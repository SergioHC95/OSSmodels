import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

root = os.path.abspath(r".\gpt-oss-20b")
print("root:", root)

tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
try:
    m = AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        use_hf_quantizer=False,   # don't activate MXFP4 even if present
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    print("✅ Loaded from ROOT shards. (These are likely non-MXFP4)")
except Exception as e:
    print("❌ Root load failed:", e)
