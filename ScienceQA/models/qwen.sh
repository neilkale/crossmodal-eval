export CUDA_VISIBLE_DEVICES=1

# Baseline
# python run_qwen.py

# Run the qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt700k
# python run_qwen.py \
#     --model qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt700k

# Run the qwen2.5_7b_150k_lora_mix_1e3_ckpt700k
# python run_qwen.py \
#     --model qwen2.5_7b_150k_lora_mix_1e3_ckpt700k 

# for checkpoint in ../../checkpoints/*; do
#     if [ -d "$checkpoint" ]; then
#         folderName=$(basename "$checkpoint")
#         if [ ! -d "../results/$folderName" ]; then
#             echo "Evaluating model from: $folderName"
#             python run_qwen.py --model "$folderName"
#         else
#             echo "Skipping $checkpoint because ../results/$folderName exists."
#         fi
#     fi
# done

folderName="qwen2.5_7b_150k_lora_blend_no_aug_plus_1e3_new_v1"
for checkpoint in ../../checkpoints/${folderName}/*; do
    if [ -d "$checkpoint" ]; then
        ckpt_name=$(basename "$checkpoint")
        if [ ! -d "../results/$folderName/$ckpt_name" ]; then
            echo "Evaluating model $ckpt_name from: $folderName"
            python run_qwen.py --model "$folderName/$ckpt_name"
        else
            echo "Skipping $checkpoint because ../results/$folderName/$ckpt_name exists."
        fi
    fi
done