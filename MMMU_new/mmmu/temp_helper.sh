CUDA_VISIBLE_DEVICES=7

folderName="qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_fix"
for checkpoint in ../../checkpoints/${folderName}/*; do
    if [ -d "$checkpoint" ]; then
        ckpt_name=$(basename "$checkpoint")
        if [ ! -f "./example_outputs/$folderName/$ckpt_name.json" ]; then
            echo "Evaluating model $ckpt_name from: $folderName"
            python run_qwen.py \
                --output_path ./example_outputs/$folderName/$ckpt_name.json \
                --model_path ../../checkpoints/$folderName/$ckpt_name
            python main_eval_only.py --output_path ./example_outputs/$folderName/$ckpt_name.json
        else
            echo "Skipping $checkpoint because ./example_outputs/$folderName/$ckpt_name.json exists."
        fi
    fi
done

# python main_eval_only.py --output_path ./example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_fix/checkpoint-300.json