CUDA_VISIBLE_DEVICES=1

folderName="qwen2.5_7b_150k_lora_blend_no_aug_plus_1e3_new_v1"
for checkpoint in ../../checkpoints/${folderName}/*; do
    if [ -d "$checkpoint" ]; then
        ckpt_name=$(basename "$checkpoint")
        if [ ! -d "./example_outputs/$folderName/$ckpt_name" ]; then
            echo "Evaluating model $ckpt_name from: $folderName"
            python run_qwen.py \
                --output_path ./example_outputs/$folderName/$ckpt_name.json \
                --model_path ../../checkpoints/$folderName/$ckpt_name
        else
            echo "Skipping $checkpoint because ./example_outputs/$folderName/$ckpt_name exists."
        fi
    fi
done