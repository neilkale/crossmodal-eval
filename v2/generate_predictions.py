import os
from helpers import get_argument_parser, get_dset_helpers
from qwen import QwenModel
import yaml

def parse_args():
    parser = get_argument_parser()
    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        config = yaml.safe_load(f)
    args.model_path = config.get('model_path')
    args.model_results_dir = config.get('results_dir')
    args.model_type = config.get('model_type')
    if not args.model_path or not args.model_results_dir or not args.model_type:
        raise ValueError("model_path, results_dir, and model_type must be specified in the config file.")
    
    args.data_dir = os.path.join('data', args.task_name.lower())
    args.output_dir = os.path.join('output', args.model_results_dir, args.task_name.lower())

    return args

if __name__ == "__main__":
    args = parse_args()
    print(f'Using model: {args.model_path}')

    if args.model_type == 'qwen2.5-vl-7b':   
        model = QwenModel(config=args.model_config)
        model_generate_func = model.query
    else:
        raise ValueError(f"Unsupported model: {args.model_path}. Currently only qwen2.5-vl-7b is supported.")
    
    print(f"Task: {args.task_name}")
    helpers = get_dset_helpers(args.task_name)

    subtasks = helpers.get_subtasks(args.subtask_name)

    for task_name in subtasks:
        print(f"Processing subtask: {task_name}")
        helpers.eval_task(task_name, model_generate_func, args.data_dir, args.output_dir, max_samples=args.max_samples, rerun=args.rerun)
    
    print("All tasks completed.")