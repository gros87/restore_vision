import torch
import gc
import os
import json
import glob
from transformers import AutoModel, AutoConfig, AutoTokenizer
from safetensors.torch import save_file
from safetensors import safe_open
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
BASE_MODEL_ID = "./Mistral-Small-3.2-24B-Instruct-2506"
FINETUNE_MODEL_ID = "./M3.2-24B-Loki-V1.3"
OUTPUT_DIR = "./M3.2-24B-Loki-V1.3-Vision-Restored"

MAX_SHARD_SIZE = 4 * 1024 * 1024 * 1024 # 4GB Shards

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. LOAD BASE
print("1. Loading Base Model (Generic)...")
config = AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
base_model = AutoModel.from_pretrained(
    BASE_MODEL_ID, config=config, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
base_sd = base_model.state_dict()

# 2. LOAD FINE-TUNE
print("2. Loading Fine-tune (Generic)...")
finetune_model = AutoModel.from_pretrained(
    FINETUNE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
ft_sd = finetune_model.state_dict()

# 3. TRANSPLANT BODY
print("3. Transplanting Body...")
ft_lookup = {}
for key in ft_sd.keys():
    core = key
    for prefix in ["language_model.model.", "language_model.", "model."]:
        if core.startswith(prefix):
            core = core.replace(prefix, "")
    ft_lookup[core] = key

replaced_count = 0
for base_key in base_sd.keys():
    base_core = base_key
    for prefix in ["language_model.model.", "language_model.", "model."]:
        if base_core.startswith(prefix):
            base_core = base_core.replace(prefix, "")
    
    if "vision_tower" in base_key or "multi_modal_projector" in base_key:
        continue 

    if base_core in ft_lookup:
        ft_real_key = ft_lookup[base_core]
        if base_sd[base_key].shape == ft_sd[ft_real_key].shape:
            base_sd[base_key] = ft_sd[ft_real_key]
            replaced_count += 1

print(f"   - Body layers overwritten: {replaced_count}")
del finetune_model, ft_sd, ft_lookup
gc.collect()

# 4. HEAD HUNTER
print("4. Hunting for Head...")
# We temporarily call it 'lm_head.weight'. The renaming step below will fix the prefix.
head_key_short = "lm_head.weight"
found_head = False

if os.path.exists(FINETUNE_MODEL_ID):
    ft_path = FINETUNE_MODEL_ID
else:
    try:
        ft_path = snapshot_download(FINETUNE_MODEL_ID, allow_patterns=["*.safetensors"])
    except:
        ft_path = "."

safetensors_files = glob.glob(os.path.join(ft_path, "*.safetensors"))

for sf_file in safetensors_files:
    try:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            keys = f.keys()
            target_key = None
            if "lm_head.weight" in keys: target_key = "lm_head.weight"
            elif "model.lm_head.weight" in keys: target_key = "model.lm_head.weight"
            elif "language_model.lm_head.weight" in keys: target_key = "language_model.lm_head.weight"
            
            if target_key:
                print(f"   >>> FOUND HEAD in: {os.path.basename(sf_file)}")
                base_sd[head_key_short] = f.get_tensor(target_key)
                found_head = True
    except:
        pass
    if found_head: break

if found_head: print("   >>> Head injected.")
else: print("   CRITICAL: Head not found.")

del base_model
gc.collect()

# =============================================================================
# 5. KEY RENAMING (The Fix for ExLlamaV3)
# =============================================================================
print("5. Renaming Keys to match Multimodal Config...")
# Mistral3ForConditionalGeneration expects 'language_model.model.layers...' structure.
# We explicitly map every key to this structure.

renamed_sd = {}
for key, tensor in base_sd.items():
    new_key = key
    
    # CASE A: Vision Components (Keep as is)
    if key.startswith("vision_tower") or key.startswith("multi_modal_projector"):
        new_key = key
        
    # CASE B: The Head
    elif "lm_head" in key:
        new_key = "language_model.lm_head.weight"
        
    # CASE C: The Text Body
    else:
        # We need to ensure it starts with 'language_model.model.'
        # Current key might be 'model.layers.0...' or just 'layers.0...' or 'embed_tokens...'
        
        clean_key = key
        # Strip existing prefixes if present to avoid doubling them
        if clean_key.startswith("model."):
            clean_key = clean_key[6:] # remove 'model.'
        if clean_key.startswith("language_model."): # Unlikely given generic load, but safest to check
            clean_key = clean_key[15:]

        new_key = f"language_model.model.{clean_key}"

    renamed_sd[new_key] = tensor

# Swap dictionaries to free memory
del base_sd
base_sd = renamed_sd
gc.collect()

# =============================================================================
# 6. SHARDED SAVING
# =============================================================================
print("6. Saving Sharded Model...")

def get_storage_size(tensor):
    return tensor.numel() * tensor.element_size()

state_dict_keys = list(base_sd.keys())
state_dict_keys.sort()

current_shard = {}
current_shard_size = 0
shard_index = 1
weight_map = {}
processed_count = 0
total_param_count = len(state_dict_keys)

for key in state_dict_keys:
    tensor = base_sd[key]
    tensor_size = get_storage_size(tensor)
    
    if current_shard_size + tensor_size > MAX_SHARD_SIZE and current_shard:
        shard_name = f"model-{shard_index:05d}.safetensors"
        print(f"     - Saving shard {shard_index}...")
        save_file(current_shard, os.path.join(OUTPUT_DIR, shard_name), metadata={"format": "pt"})
        del current_shard
        current_shard = {}
        current_shard_size = 0
        shard_index += 1
        gc.collect()

    current_shard[key] = tensor
    current_shard_size += tensor_size
    weight_map[key] = shard_index 
    base_sd[key] = None 
    processed_count += 1
    if processed_count % 100 == 0: print(f"       Processed {processed_count}/{total_param_count}...")

if current_shard:
    shard_name = f"model-{shard_index:05d}.safetensors"
    print(f"     - Saving final shard {shard_index}...")
    save_file(current_shard, os.path.join(OUTPUT_DIR, shard_name), metadata={"format": "pt"})

total_shards = shard_index

# =============================================================================
# 7. FINALIZE
# =============================================================================
print("7. Finalizing...")
final_weight_map = {}
for i in range(1, total_shards + 1):
    old_name = f"model-{i:05d}.safetensors"
    new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
    os.rename(os.path.join(OUTPUT_DIR, old_name), os.path.join(OUTPUT_DIR, new_name))
    for key, shard_id in weight_map.items():
        if shard_id == i: final_weight_map[key] = new_name

index_data = {"metadata": {"total_size": 0}, "weight_map": final_weight_map}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index_data, f, indent=2)

base_config_dict = config.to_dict()
base_config_dict["architectures"] = ["Mistral3ForConditionalGeneration"]
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(base_config_dict, f, indent=2)

tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL_ID)
tokenizer.save_pretrained(OUTPUT_DIR)

try:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.save_pretrained(OUTPUT_DIR)
except: pass

print(f"Done! Saved to {OUTPUT_DIR}")