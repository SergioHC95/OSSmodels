import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parent_dir = os.path.abspath(r".\gpt-oss-20b")
weights_subfolder = r"original\bf16"  # or r"original\fp32"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(parent_dir, local_files_only=True)

print("Loading model (this will use ~40 GB RAM for bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    parent_dir,
    subfolder=weights_subfolder,
    local_files_only=True,
    torch_dtype=torch.float32,  # safe for CPU, will auto-convert
    device_map="cpu",
    low_cpu_mem_usage=True,
    attn_implementation="eager",
    use_hf_quantizer=False
)

prompt = "Explain the bias-variance tradeoff in two sentences."
inputs = tok(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
