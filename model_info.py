from transformer_lens import HookedTransformer

print("[*] Initializing model configuration inspection...")
model = HookedTransformer.from_pretrained("pythia-160m", device="cpu")

print("\n==============================================")
print("⚙️  ELEUTHERAI PYTHIA-160M ARCHITECTURE BLUEPRINT")
print("==============================================")
print(f"• Total Block Layers (Depth)       : {model.cfg.n_layers}")
print(f"• Hidden Dimension Width (d_model)  : {model.cfg.d_model}")
print(f"• Attention Heads per Layer        : {model.cfg.n_heads}")
print(f"• Head Dimension Size (d_head)     : {model.cfg.d_head}")
print(f"• MLP Layer Width (Neurons)        : {model.cfg.d_mlp}")
print(f"• Max Context Window Length        : {model.cfg.n_ctx}")
print(f"• Total Vocab Size Size            : {model.cfg.d_vocab}")
print("==============================================\n")