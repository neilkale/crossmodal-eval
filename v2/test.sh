export CUDA_VISIBLE_DEVICES=4,5,6,7
# python generate_predictions.py --model_config configs/qwen_base_attn0.0.yaml
# python evaluate_predictions.py --model_config configs/qwen_base_attn0.0.yaml
python summary_stats.py --model_config configs/qwen_base_attn0.0.yaml