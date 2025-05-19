## For inference
CUDA_VISIBLE_DEVICES=7

# python test_benchmark.py --task_name='all' --model_name='QWEN' --model_path='Qwen/Qwen2.5-VL-7B-Instruct'

# python test_benchmark.py --task_name='all' --model_name='QWEN-HIT' --model_path='qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt700k'

# python test_benchmark.py --task_name='all' --model_name='QWEN-IT' --model_path='qwen2.5_7b_150k_lora_mix_1e3_ckpt700k'

# python eval/test_benchmark.py --model_name='QWEN-HIT-PLUS' --task_name='all' --model_path='qwen2.5_7b_150k_lora_blend_no_aug_plus_1e3_new_v1_ckpt200k' 

# python test_benchmark.py --task_name='all' --model_name='QWEN-HIT-FIX' --model_path='qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_fix/checkpoint-200'

folderName="qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_fix"
checkpoint_base="../../checkpoints/${folderName}"
output_dir="./outputs"

for checkpoint in "${checkpoint_base}"/*; do
    # skip anything that isn’t a directory
    [ -d "${checkpoint}" ] || continue

    ckpt_name="$(basename "${checkpoint}")"
    model_name="${folderName}_${ckpt_name}"
    output_file="${output_dir}/${model_name}.json"

    if [ ! -f "${output_file}" ]; then
        echo "Evaluating model ${ckpt_name} from ${folderName}…"
        python test_benchmark.py \
            --task_name all \
            --model_name "${model_name}" \
            --model_path "${checkpoint}"
    else
        echo "Skipping ${ckpt_name} because ${output_file} exists."
    fi
done


## For scores (set the models to be evaluated in `model_names` array in `evaluate.py`)

python evaluate.py