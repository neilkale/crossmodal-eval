#!/bin/bash

# 700k no aug new v1 with weight ensemble
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt700k_wise1.5_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt700k \
--weight_ensembling_ratio 1.5

# 60k no aug new v1
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt60k_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt60k

# 5k no aug
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt5k_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt5k

# 10k no aug
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt10k_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt10k

# 100k no aug
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt100k_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt100k

# 400k no aug
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt400k_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt400k

# 400k
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_1e3_ckpt400k_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_1e3_ckpt400k

# 800k no aug
python run_qwen.py \
--output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt800k_val.json \
--model_path ../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_ckpt800k

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