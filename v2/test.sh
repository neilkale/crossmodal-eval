export CUDA_VISIBLE_DEVICES=4,5,6,7
python generate_predictions.py --max_samples 10
python evaluate_predictions.py
