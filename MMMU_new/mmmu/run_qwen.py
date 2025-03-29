import torch
import os
import random

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from qwen_vl_utils import process_vision_info
from utils.qwen_model import QwenModel

from argparse import ArgumentParser

from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG, convert_sample_to_interleaved
from utils.eval_utils import parse_multi_choice_response, parse_open_response

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_model(args, samples, model):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples, desc='running model'):
            interleaved_text_images = convert_sample_to_interleaved(sample)

            response = model.generate(
                interleaved_text_images,
                system_prompt = None,
            )[0]

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans

    return out_samples

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='example_outputs/qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt60k_val.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/qwen2.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_path', type=str, default="../../qwen2.5_7b_150k_lora_blend_no_aug_1e3_new_v1_ckpt60k")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_ensembling_ratio', type=float, default=1)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # Initialize and load model
    print('qwen2.5_initializing...')
    model = QwenModel(model=args.model_path, weight_ensembling_ratio=args.weight_ensembling_ratio)
    
    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in tqdm(CAT_SHORT2LONG.values(), desc='loading dataset'):
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    samples = []
    for sample in tqdm(dataset, desc='preprocessing dataset'):
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)

        if sample['image']:
            
            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": sample['image']},
                    {"type": "text", "text": sample['question']}
                ]
            }]

            sample['image'], _ = process_vision_info(message)

        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model)

    save_json(args.output_path, out_samples)


if __name__ == '__main__':
    main()