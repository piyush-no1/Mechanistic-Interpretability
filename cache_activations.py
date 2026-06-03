import os
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
from safetensors.torch import save_file

LAYER = 6  
HOOK_POINT = f"blocks.{LAYER}.hook_resid_post"
MAX_TOKENS = 10_000_000 
CONTEXT_LENGTH = 256 
BATCH_SIZE = 32
SAVE_CHUNK_SIZE = 500_000  
OUTPUT_DIR = "activation_cache"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Python 3.12 Production Pipeline Active. Device: {device}")

print("[*] Initializing Pythia-160M into TransformerLens...")
model = HookedTransformer.from_pretrained("pythia-160m", device=device)

print("[*] Consolidating text corpus into local workspace...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=False)
print(f"[+] Local dataset available. Total text sequences: {len(dataset)}")

collected_tokens = 0
activation_pool = []
file_counter = 0
current_batch_texts = []

print("[*] Extracting intermediate hidden states. Processing 10M tokens...")
for item in tqdm(dataset):
    text = item["text"].strip()
    if len(text) < 50: 
        continue
    
    current_batch_texts.append(text)
    
    if len(current_batch_texts) == BATCH_SIZE:
        tokens = model.to_tokens(current_batch_texts, truncate=True, prepend_bos=True)
        
        if tokens.shape[1] < CONTEXT_LENGTH:
            tokens = torch.nn.functional.pad(tokens, (0, CONTEXT_LENGTH - tokens.shape[1]), value=model.tokenizer.pad_token_id)
        else:
            tokens = tokens[:, :CONTEXT_LENGTH]
            
        tokens = tokens.to(device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_POINT])
            hidden_states = cache[HOOK_POINT] 
            
            flat_activations = hidden_states.reshape(-1, hidden_states.shape[-1]).cpu()
            activation_pool.append(flat_activations)
            collected_tokens += flat_activations.shape[0]
            
        current_batch_texts = []
        
        if collected_tokens >= SAVE_CHUNK_SIZE:
            save_tensor = torch.cat(activation_pool, dim=0)[:SAVE_CHUNK_SIZE]
            output_path = os.path.join(OUTPUT_DIR, f"base_layer_{LAYER}_chunk_{file_counter}.safetensors")
            save_file({"activations": save_tensor}, output_path)
            
            file_counter += 1
            activation_pool = []
            collected_tokens = 0
            
    if file_counter * SAVE_CHUNK_SIZE >= MAX_TOKENS:
        print(f"\n[✔] Milestone Achieved: {MAX_TOKENS:,} activations saved cleanly across {file_counter} chunks!")
        break