import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformer_lens import HookedTransformer

D_MODEL = 768
D_SAE = 7680
LAYER = 6
HOOK_POINT = f"blocks.{LAYER}.hook_resid_post"
BASE_SAE_PATH = "sae_baseline_10x.pt"
FT_SAE_PATH = "sae_finetuned_10x.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Initializing Qualitative Inspection Engine on Hardware: {device}")

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model=768, d_sae=7680): 
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae))
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model)) 

    def forward(self, x):
        x_centered = x - self.decoder_bias
        latent_acts = torch.relu(self.encoder(x_centered) + self.encoder_bias)
        return latent_acts

sae_base = SparseAutoencoder(D_MODEL, D_SAE).to(device)
sae_base.load_state_dict(torch.load(BASE_SAE_PATH, map_location=device))

sae_ft = SparseAutoencoder(D_MODEL, D_SAE).to(device)
sae_ft.load_state_dict(torch.load(FT_SAE_PATH, map_location=device))

print("[*] Compiling Base and Fine-Tuned Transformer Models...")
model_base = HookedTransformer.from_pretrained("pythia-160m", device=device)

hf_base = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
hf_peft = PeftModel.from_pretrained(hf_base, "pythia_160m_python_finetuned")
hf_merged = hf_peft.merge_and_unload()
model_ft = HookedTransformer.from_pretrained("pythia-160m", hf_model=hf_merged, device=device)

test_prompt = "The developer wrote a function to sort arrays: def sort_data(x): return sorted(x)"

print("[*] Probing hidden representations token-by-token...")
tokens_base = model_base.to_tokens(test_prompt).to(device)
tokens_ft = model_ft.to_tokens(test_prompt).to(device)

str_tokens = model_base.to_str_tokens(test_prompt)

with torch.no_grad():
    _, cache_base = model_base.run_with_cache(tokens_base, names_filter=[HOOK_POINT])
    acts_base = cache_base[HOOK_POINT][0] # Isolate batch element 0 -> [seq_len, 768]
    latent_acts_base = sae_base(acts_base) # Map to sparse space -> [seq_len, 7680]

    _, cache_ft = model_ft.run_with_cache(tokens_ft, names_filter=[HOOK_POINT])
    acts_ft = cache_ft[HOOK_POINT][0]
    latent_acts_ft = sae_ft(acts_ft)

print("\n" + "="*85)
print(f"🔬 MICROSCOPIC FEATURE ACTIVATION REPORT FOR PROMPT:")
print(f"'{test_prompt}'")
print("="*85)
print(f"{'Token':<15} | {'Top Baseline Feature (Idx: Val)':<32} | {'Top Fine-Tuned Feature (Idx: Val)':<32}")
print("-"*85)

for i, token_str in enumerate(str_tokens):
    val_base, idx_base = torch.max(latent_acts_base[i], dim=0)
    val_ft, idx_ft = torch.max(latent_acts_ft[i], dim=0)
    
    base_str = f"#{idx_base.item():04d} ({val_base.item():.2f})" if val_base.item() > 0.0 else "None (0.00)"
    ft_str = f"#{idx_ft.item():04d} ({val_ft.item():.2f})" if val_ft.item() > 0.0 else "None (0.00)"
    
    clean_token = token_str.replace("\n", "\\n").replace(" ", "·")
    print(f"{clean_token:<15} | {base_str:<32} | {ft_str:<32}")

print("="*85 + "\n")