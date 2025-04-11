## For inference

python test_benchmark.py --task_name='all' --model_name='QWEN' --model_path='Qwen/Qwen2.5-VL-7B-Instruct'

python test_benchmark.py --task_name='all' --model_name='QWEN-HIT' --model_path='qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt700k'

python test_benchmark.py --task_name='all' --model_name='QWEN-IT' --model_path='qwen2.5_7b_150k_lora_mix_1e3_ckpt700k'

## For scores (set the models to be evaluated in `model_names` array in `evaluate.py`)

python evaluate.py