import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import load_file
from tqdm import tqdm
import matplotlib.pyplot as plt

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model=768, d_sae=7680): 
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        
        self.encoder = nn.Linear(d_model, d_sae)
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae))
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model)) 

    def forward(self, x):
        x_centered = x - self.decoder_bias
        latent_acts = torch.relu(self.encoder(x_centered) + self.encoder_bias)
        reconstruction = self.decoder(latent_acts) + self.decoder_bias
        return reconstruction, latent_acts

def train_fine_tuned_sae(
    activation_cache_dir="activation_cache_ft",
    baseline_weights_path="sae_baseline_10x.pt",
    save_weights_path="sae_finetuned_10x.pt",
    epochs=3,                  
    batch_size=4096,
    l1_coeff=1e-4,             
    lr=8e-5,                  
    proximal_alpha=0.001      
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(f"[*] Hyperparameters Adjusted. Fine-Tuning SAE Active on: {device}")
    print("="*70 + "\n")

    sae = SparseAutoencoder(d_model=768, d_sae=7680).to(device)
    
    print(f"[*] Loading baseline SAE weights from {baseline_weights_path}...")
    try:
        baseline_state = torch.load(baseline_weights_path, map_location=device)
        sae.load_state_dict(baseline_state)
        print("[+] Successfully warm-started fine-tuned SAE with baseline weights.")
    except Exception as e:
        print(f"[X] Error loading baseline weights checkpoint: {e}")
        return

    base_enc_w = baseline_state["encoder.weight"].to(device).clone().detach()
    base_dec_w = baseline_state["decoder.weight"].to(device).clone().detach()

    cache_files = [f for f in os.listdir(activation_cache_dir) if f.endswith('.safetensors') and "ft_" in f]
    if not cache_files:
        print(f"[X] FATAL ERROR: No fine-tuned activation chunk files found in '{activation_cache_dir}'!")
        return
        
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    metrics_history = {"step": [], "loss": [], "mse": [], "prox_loss": [], "dead_features": []}
    global_step = 0
    
    print("[*] Adapting latent concept directions with soft proximity protections...")
    for epoch in range(1, epochs + 1):
        random.shuffle(cache_files)
        file_progress = tqdm(cache_files, desc=f"Epoch {epoch}/{epochs}")
        
        feature_activation_counts = torch.zeros(sae.d_sae, device=device)
        
        for filename in file_progress:
            file_path = os.path.join(activation_cache_dir, filename)
            
            try:
                chunk_data = load_file(file_path)
                activations = chunk_data["activations"].to(torch.float32)
            except Exception as e:
                print(f"\n[!] Skipping unloadable file {filename}: {e}")
                continue
                
            dataset = TensorDataset(activations)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for batch in dataloader:
                x = batch[0].to(device)
                
                reconstruction, latent_acts = sae(x)
                
                mse_loss = nn.functional.mse_loss(reconstruction, x)
                l1_loss = torch.norm(latent_acts, p=1, dim=-1).mean()
                
                # Proximal alignment loss calculation
                enc_proximal = nn.functional.mse_loss(sae.encoder.weight, base_enc_w)
                dec_proximal = nn.functional.mse_loss(sae.decoder.weight, base_dec_w)
                prox_loss = proximal_alpha * (enc_proximal + dec_proximal)
                
                loss = mse_loss + (l1_coeff * l1_loss) + prox_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    sae.decoder.weight.data = nn.functional.normalize(sae.decoder.weight.data, dim=0)
                
                feature_activation_counts += (latent_acts > 0).sum(dim=0)
                
                if global_step % 20 == 0:
                    current_dead = (feature_activation_counts == 0).sum().item()
                    metrics_history["step"].append(global_step)
                    metrics_history["loss"].append(loss.item())
                    metrics_history["mse"].append(mse_loss.item())
                    metrics_history["prox_loss"].append(prox_loss.item())
                    metrics_history["dead_features"].append(current_dead)

                global_step += 1
                
            file_progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MSE": f"{mse_loss.item():.4f}",
                "Prox": f"{prox_loss.item():.4f}"
            })
            
            del activations
            del chunk_data
            torch.cuda.empty_cache()
            
    torch.save(sae.state_dict(), save_weights_path)
    print(f"\n[+] Aligned weights checkpoint saved cleanly to: {save_weights_path}")
    
    # Generate charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(metrics_history["step"], metrics_history["loss"], label="Total Combined Loss", color="tab:red", alpha=0.5)
    ax1.plot(metrics_history["step"], metrics_history["mse"], label="Reconstruction MSE", color="tab:orange", linestyle="--")
    ax1.plot(metrics_history["step"], metrics_history["prox_loss"], label="Proximal Protection", color="tab:green", linestyle=":")
    ax1.set_title("Proximal Anchored Training Path", fontweight="bold")
    ax1.set_xlabel("Optimization Step")
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    
    ax2.plot(metrics_history["step"], metrics_history["dead_features"], color="tab:blue")
    ax2.set_title("Active Feature Density Variance", fontweight="bold")
    ax2.set_xlabel("Optimization Step")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("sae_ft_training_metrics.png", dpi=300)
    print("[✔] Balanced diagnostic report plots generated successfully.")

if __name__ == "__main__":
    train_fine_tuned_sae()