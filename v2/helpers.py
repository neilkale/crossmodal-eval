import argparse
import os
import yaml
from hardblink import HardBLINK
from blink import BLINK

def get_task(task_name, data_dir=None, output_dir=None):
    if task_name == 'HardBLINK':
        return HardBLINK(data_dir=data_dir, output_dir=output_dir)
    elif task_name == 'BLINK':
        return BLINK(data_dir=data_dir, output_dir=output_dir)
    else:
        raise ValueError(f"Unsupported task: {task_name}.")

def get_argument_parser():
    parser = argparse.ArgumentParser(description="Base arguments for evaluation.")

    parser.add_argument(
        "--model_config",
        type=str,
        default='configs/qwen_base_attn0.0.yaml',
        help="Path to the model config.",
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default='HardBLINK',
        help='Task name to be evaluated.'
    )
    parser.add_argument(
        '--subtask_name',
        type=str,
        default='all',
        help='Subtask name to be evaluated.'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process.'
    )
    parser.add_argument(
        '--rerun',
        action='store_true',
        default=False,
        help='Rerun the evaluation even if results already exist.'
    )

    return parser

