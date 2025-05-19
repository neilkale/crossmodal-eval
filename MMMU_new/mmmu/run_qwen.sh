#!/bin/bash

# Useful scripts for evaluating the 'coreasoning prompt' models.
mv /work3/nkale/ml-projects/understanding-conflict/Qwen2-VL-Finetune/output/llava_instruct_150k_lora_blend_no_aug_1e3_new_v1_fix/checkpoint-200/ ../../checkpoints/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_fix/checkpoint-200/

python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_fix4/checkpoint-100.json \
--model_path ../../checkpoints/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_fix4/checkpoint-100

python main_eval_only.py --output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_fix4/checkpoint-300.json

# 700k no aug new v1 with weight ensemble
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt700k_wise1.5_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt700k \
--weight_ensembling_ratio 1.5

# 1100k no aug with weight ensemble
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt1100k_wise0.05_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt1100k \
--weight_ensembling_ratio 0.05

python main_eval_only.py --output_path ./example_outputs/{INSERT PATH HERE}

# Baseline
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b.json \
--model_path "Qwen/Qwen2.5-VL-7B-Instruct"

python main_eval_only.py --output_path ./example_outputs/qwen2.5_7b.json