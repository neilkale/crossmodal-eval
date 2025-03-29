import json
from utils.eval_utils import parse_open_response

in_file = "example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt400k_val.json"
out_file = "example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt400k_val_fixed.json"

with open(in_file, "r") as f:
    data = json.load(f)

for key, value in data.items():
    if value not in ["A", "B", "C", "D", "E"]:
        data[key] = str(parse_open_response(value)[0])

with open(out_file, "w") as f:
    json.dump(data, f, indent=2)