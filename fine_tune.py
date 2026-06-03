import os
import torch
import inspect
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("\n" + "="*70)
print("[*] Launching Fine-Tuning Pipeline...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Target Compute Hardware: {device}")
print("="*70 + "\n")

MODEL_ID = "EleutherAI/pythia-160m"
OUTPUT_DIR = "pythia_160m_python_finetuned"
PLOT_OUTPUT_PATH = "ft_training_loss.png"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map={"": device})

peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["query_key_value", "dense"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("[*] Downloading stable Python instruction dataset")
raw_dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2500]")

print("[*] Pre-compiling raw instruct pairs into an explicit 'text' layer...")
def compile_alpaca_rows(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = raw_dataset.map(compile_alpaca_rows, remove_columns=raw_dataset.column_names)

sft_config_params = inspect.signature(SFTConfig.__init__).parameters
sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters

config_kwargs = {
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "logging_steps": 20,
    "num_train_epochs": 1,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "fp16": True,
    "save_strategy": "no",
    "report_to": "none",
}

trainer_kwargs = {
    "model": model,
    "train_dataset": dataset,
}

if "max_length" in sft_config_params:
    config_kwargs["max_length"] = 256
elif "max_seq_length" in sft_config_params:
    config_kwargs["max_seq_length"] = 256

if "dataset_text_field" in sft_config_params:
    config_kwargs["dataset_text_field"] = "text"
else:
    trainer_kwargs["dataset_text_field"] = "text"

if "packing" in sft_config_params:
    config_kwargs["packing"] = False
else:
    trainer_kwargs["packing"] = False

if "processing_class" in sft_trainer_params:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

training_args = SFTConfig(**config_kwargs)
trainer_kwargs["args"] = training_args

trainer = SFTTrainer(**trainer_kwargs)

print("[*] Training on Python corpus. Optimizing...")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n[✔] Fine-Tuning Complete! Adapters saved cleanly to -> {OUTPUT_DIR}")


print("[*] Training finished. Extracting history logs for graphing...")
history = trainer.state.log_history

steps = []
losses = []
for entry in history:
    if "loss" in entry:
        steps.append(entry["step"])
        losses.append(entry["loss"])

if steps:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Fine-Tuning Loss", color="tab:orange", linewidth=2)
    plt.xlabel("Optimization Steps (Log Intervals)", fontweight="bold")
    plt.ylabel("Cross-Entropy Loss Value", fontweight="bold")
    plt.title("Pythia-160M Python Adaptation Loss Trajectory", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    print(f"[✔] Technical graph asset generated successfully -> {PLOT_OUTPUT_PATH}")
else:
    print("[!] Warning: No loss metrics were captured in log history to generate a plot.")