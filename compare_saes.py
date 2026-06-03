import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

D_MODEL = 768
D_SAE = 7680  
BASE_SAE_PATH = "sae_baseline_10x.pt"
FT_SAE_PATH = "sae_finetuned_10x.pt"
HIST_OUTPUT_PATH = "feature_drift_histogram.png"
HEAT_OUTPUT_PATH = "cross_alignment_heatmap.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n" + "="*70)
print("[*] Launching Phase 4: Cross-SAE Comparative Analytics System...")
print(f"[*] Active Computation Hardware: {device}")
print("="*70 + "\n")

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae))
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))

print("[*] Loading baseline SAE weights matrix...")
sae_base = SparseAutoencoder(D_MODEL, D_SAE).to(device)
sae_base.load_state_dict(torch.load(BASE_SAE_PATH, map_location=device))

print("[*] Loading fine-tuned Python SAE weights matrix...")
sae_ft = SparseAutoencoder(D_MODEL, D_SAE).to(device)
sae_ft.load_state_dict(torch.load(FT_SAE_PATH, map_location=device))

W_dec_base = sae_base.decoder.weight.data
W_dec_ft = sae_ft.decoder.weight.data

norm_base = W_dec_base / W_dec_base.norm(dim=0, keepdim=True)
norm_ft = W_dec_ft / W_dec_ft.norm(dim=0, keepdim=True)

print("[*] Computing global 7,680 x 7,680 cross-alignment similarity matrix...")
with torch.no_grad():
    similarity_matrix = torch.mm(norm_base.t(), norm_ft).cpu()

print("[*] Resolving peak feature drift convergence paths...")
max_similarities, best_ft_indices = torch.max(similarity_matrix, dim=1)
max_sim_np = max_similarities.numpy()

print("\n=======================================================")
print("📊 PREVIEW METRIC REPORT: LATENT COGNITIVE SHIFTS")
print("=======================================================")
for base_idx in range(10):
    print(f"• Base Feature #{base_idx:03d} -> Best Post-SFT Match #{best_ft_indices[base_idx].item():03d} | Max CosSim: {max_sim_np[base_idx]:.4f}")
print("=======================================================\n")

print("[*] Generating Graph 1: Feature Drift Max-Cosine Similarity Histogram...")
plt.figure(figsize=(10, 5.5))
n, bins, patches = plt.hist(max_sim_np, bins=60, color='tab:purple', alpha=0.75, edgecolor='black', linewidth=0.5)

plt.axvline(x=0.3, color='tab:red', linestyle=':', alpha=0.7, label='Extinction Boundary')
plt.axvline(x=0.7, color='tab:green', linestyle=':', alpha=0.7, label='Invariant Threshold')

plt.xlabel('Maximum Cosine Similarity Vector Coordinate ($W_{base} \cdot W_{ft}$)', fontweight='bold')
plt.ylabel('Latent Feature Frequency Count (Dictionary Density)', fontweight='bold')
plt.title('Distribution of Latent Feature Drift Across Domain Shifts (10M Tokens)', fontsize=12, fontweight='bold', pad=12)
plt.grid(True, alpha=0.2, linestyle=':')
plt.xlim(0.0, 1.0)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(HIST_OUTPUT_PATH, dpi=300)
print(f"[✔] Global distribution graph exported successfully -> {HIST_OUTPUT_PATH}")

print("[*] Generating Graph 2: Microscopic Cross-Alignment Submatrix Heatmap...")
plt.figure(figsize=(8, 7))

slice_size = 40
similarity_slice = similarity_matrix[:slice_size, :slice_size].numpy()

im = plt.imshow(similarity_slice, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
cbar = plt.colorbar(im)
cbar.set_label('Absolute Vector Alignment Magnitude', fontweight='bold')

plt.xlabel('Fine-Tuned Dictionary Sub-Space (Features 0-40)', fontweight='bold')
plt.ylabel('Baseline Dictionary Sub-Space (Features 0-40)', fontweight='bold')
plt.title('Cross-Coding Matrix Alignment Microscale Footprint', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(HEAT_OUTPUT_PATH, dpi=300)
print(f"[✔] Matrix cross-alignment heatmap exported successfully -> {HEAT_OUTPUT_PATH}")

print("\n[✔] Phase 4 complete! All structural evaluation metrics generated cleanly.")