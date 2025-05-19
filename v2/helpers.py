import argparse
import os
import yaml
import hardblink_helpers
import blink_helpers

HELPERS = {
    'HardBLINK': hardblink_helpers,
    'BLINK':    blink_helpers,
}

def get_dset_helpers(task_name):
    if task_name == 'HardBLINK':
        return hardblink_helpers
    elif task_name == 'BLINK':
        return blink_helpers
    else:
        raise ValueError(f"Unsupported task: {task_name}.")

def get_argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate predictions for HardBLINK.")

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