cd ../evaluation

##### qwen-2.5-vl-7b-instruct #####
# generate solution
python generate_response.py \
--model_base qwen \
--model_path qwen2.5_7b_150k_lora_mix_1e3_ckpt700k \
--output_dir ../results/qwen2.5_7b_150k_lora_mix_1e3_ckpt700k \
--output_file output_qwen.json

# extract answer
python extract_answer.py \
--results_file_path ../results/qwen2.5_7b_150k_lora_mix_1e3_ckpt700k/output_qwen.json 

# calculate score
python calculate_score.py \
--output_dir ../results/qwen2.5_7b_150k_lora_mix_1e3_ckpt700k \
--output_file output_qwen.json \
--score_file scores_qwen.json
