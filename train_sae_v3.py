import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import load_file
from tqdm import tqdm
import matplotlib.pyplot as plt


D_MODEL = 768  
EXPANSION_FACTOR = 10  
D_SAE = 7680  
L1_COEFF = 0.0008  
BATCH_SIZE = 4096  
LEARNING_RATE = 0.0003
EPOCHS = 5
INPUT_DIR = "activation_cache"
MODEL_OUTPUT_PATH = "sae_baseline_10x.pt"
PLOT_OUTPUT_PATH = "sae_training_metrics.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n" + "="*70)
print(f"[*] Execution Hardware Device Detected: {device}")
print("="*70 + "\n")

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae))
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        
        init_weights = torch.randn(d_model, d_sae)
        init_weights = init_weights / init_weights.norm(dim=0, keepdim=True)
        self.decoder.weight.data = init_weights

    def forward(self, x):
        x_centered = x - self.decoder_bias
        feature_acts = torch.relu(self.encoder(x_centered) + self.encoder_bias)
        x_reconstructed = self.decoder(feature_acts) + self.decoder_bias
        return x_reconstructed, feature_acts

safetensor_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".safetensors") and "chunk_" in f]
if len(safetensor_files) == 0:
    print("[X] FATAL ERROR: No valid activation chunk files found in 'activation_cache'!")
    exit(1)

sae = SparseAutoencoder(d_model=D_MODEL, d_sae=D_SAE).to(device)
optimizer = optim.Adam(sae.parameters(), lr=LEARNING_RATE)

metrics_history = {"step": [], "loss": [], "mse": [], "sparsity": []}
global_step = 0

print(f"[*] Starting Streaming Optimization Loop. Dictionary Width: {D_SAE} Features.")
for epoch in range(1, EPOCHS + 1):
    random.shuffle(safetensor_files)
    
    file_progress = tqdm(safetensor_files, desc=f"Epoch {epoch}/{EPOCHS}")
    for filename in file_progress:
        file_path = os.path.join(INPUT_DIR, filename)
        
        try:
            chunk_data = load_file(file_path)
            activations = chunk_data["activations"].to(torch.float32)
        except Exception as e:
            print(f"\n[!] Skipping un-loadable file {filename}: {e}")
            continue
            
        dataset = TensorDataset(activations)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            x_reconstruct, feature_acts = sae(x)
            mse_loss = nn.functional.mse_loss(x_reconstruct, x, reduction="mean")
            l1_loss = torch.linalg.norm(feature_acts, ord=1, dim=-1).mean()
            loss = mse_loss + (L1_COEFF * l1_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                sae.decoder.weight.data = sae.decoder.weight.data / sae.decoder.weight.data.norm(dim=0, keepdim=True)
                
            dead_features_pct = (feature_acts == 0).float().mean().item() * 100.0
            
            if global_step % 20 == 0:
                metrics_history["step"].append(global_step)
                metrics_history["loss"].append(loss.item())
                metrics_history["mse"].append(mse_loss.item())
                metrics_history["sparsity"].append(dead_features_pct)

            global_step += 1
            
        file_progress.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "MSE": f"{mse_loss.item():.4f}",
            "Sparsity": f"{dead_features_pct:.1f}%"
        })
        
        del activations
        del chunk_data
        torch.cuda.empty_cache()

torch.save(sae.state_dict(), MODEL_OUTPUT_PATH)
print(f"\n[+] Weights saved cleanly to: {MODEL_OUTPUT_PATH}")

print("[*] Rendering training analytics visualization...")
fig, ax1 = plt.subplots(figsize=(11, 6))
color_loss = 'tab:red'
ax1.set_xlabel('Optimization Iteration Step (Every 20 Batches)', fontweight='bold')
ax1.set_ylabel('Loss / Mean Squared Error Value', color=color_loss, fontweight='bold')
curve_loss = ax1.plot(metrics_history["step"], metrics_history["loss"], color=color_loss, alpha=0.5, label='Total Objective Loss')
curve_mse = ax1.plot(metrics_history["step"], metrics_history["mse"], color='tab:orange', linestyle='--', alpha=0.8, label='Reconstruction MSE')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.grid(True, alpha=0.25, linestyle=':')

ax2 = ax1.twinx()  
color_sparse = 'tab:blue'
ax2.set_ylabel('Latent Feature Sparsity Percentage (%)', color=color_sparse, fontweight='bold')
curve_sparse = ax2.plot(metrics_history["step"], metrics_history["sparsity"], color=color_sparse, alpha=0.7, label='Sparsity Level (%)')
ax2.tick_params(axis='y', labelcolor=color_sparse)

all_curves = curve_loss + curve_mse + curve_sparse
all_labels = [c.get_label() for c in all_curves]
ax1.legend(all_curves, all_labels, loc='center right', frameon=True, facecolor='#f7f7f7')
plt.title('10x Sparse Autoencoder Loss Convergence & Sparsity Trajectory (Streaming Chunks)', fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
print(f"[✔] Technical graph asset generated successfully -> {PLOT_OUTPUT_PATH}")