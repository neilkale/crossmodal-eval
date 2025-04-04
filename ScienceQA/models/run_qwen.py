import os
import re
import json
import argparse
import random
from tqdm import tqdm
from base_prompt import *
from qwen_model import QwenModel
from qwen_vl_utils import process_vision_info

def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    qids = pid_splits['%s' % (args.test_split)]
    qids = qids[:args.test_number] if args.test_number > 0 else qids
    print(f"number of test problems: {len(qids)}\n")

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    train_qids = pid_splits['train']
    if shot_qids == None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        shot_qids = random.sample(train_qids, args.shot_number)  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids  # check shot_qids
    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, qids, shot_qids

def process_images(image_paths):
    if not image_paths:
        return []
    message = [{
                "role": "user",
                "content": 
                    [{
                        "type": "image",
                        "image": "file://" + image_path,
                    }]
            } for image_path in image_paths]
    images, _ = process_vision_info(message)
    return images

def convert_prompt_to_qwen_format(prompt, images):
    # We assume that the prompt contains one or more occurrences of the image tag "<image>"
    # Split the prompt on the image tag.
    parts = prompt.split("<IMAGES>")
    content_list = []

    # For every split part, add the text then, if available, an image entry
    for i, text_part in enumerate(parts):
        # Append the text part (if non-empty)
        if text_part.strip():
            content_list.append({"type": "text", "text": text_part.strip()})
        # If there's an image placeholder after this text part, pop one image from img_list
        if i < len(parts) - 1:
            for img in images:
                content_list.append({"type": "image", "image": img})   

    return content_list

def get_qwen_result(model, prompt, images, args):
    
    prompt = convert_prompt_to_qwen_format(prompt, images)

    responses = model.generate(
                prompt,
                system_prompt = None)
    output = responses[0]

    # extract the answer
    pattern = re.compile(r'The answer is ([A-Z]).')
    res = pattern.findall(output)
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"

    return answer, output


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def get_result_file(args):
    result_file = "{}/{}/{}_{}_{}_{}_{}_seed_{}.json".format(args.output_root, args.model, args.weight_ensembling_ratio, args.label, args.test_split, args.prompt_format, args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, shot_qids, args, results, outputs):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['shot_qids'] = shot_qids
    data['args'] = vars(args)
    data['results'] = results
    data['outputs'] = outputs

    if not os.path.exists(result_file):
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/scienceqa')
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--caption_file', type=str, default='../data/captions.json')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--weight_ensembling_ratio', type=float, default=1)
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_number', type=int, default=-1, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
                        ],
                        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=3, help='Number of n-shot training examples.')
    parser.add_argument('--shot_qids', type=list, default=None, help='Question indexes of shot examples')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    problems, qids, shot_qids = load_data(args)  # probelms, test question ids, shot example ids

    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        acc = check_point['acc']
        correct = check_point['correct']
        results = check_point['results']
        outputs = check_point['outputs']
        print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%")
    else:
        correct = 0
        results = {}
        outputs = {}

    # load the model
    print('qwen2.5_initializing...')
    if args.model == 'Qwen/Qwen2.5-VL-7B-Instruct':
        model = QwenModel(model=args.model, weight_ensembling_ratio=args.weight_ensembling_ratio)
    else:
        model = QwenModel(model='../../checkpoints/' + args.model, weight_ensembling_ratio=args.weight_ensembling_ratio)

    # for qid in tqdm(qids):
    for i, qid in tqdm(enumerate(qids), total=len(qids)):
        if qid in results:
            continue

        choices = problems[qid]["choices"]
        answer = problems[qid]["answer"]  # 0, 1, ..., 4
        label = args.options[answer]  # 'A', ..., 'E'

        # generate prompt
        prompt = build_prompt(problems, shot_qids, qid, args)

        # get images 
        img_dir = os.path.join(args.data_root, problems[qid]["split"], qid)
        if os.path.exists(img_dir):
            img_paths = [os.path.join(img_dir, f)
                    for f in os.listdir(img_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'image' in f.lower()]
            images = process_images(img_paths)
        else:
            images = []
        

        # generate prediction
        prediction, output = get_qwen_result(model, prompt, images, args)  # 'A', ..., 'E'
        pred_idx = get_pred_idx(prediction, choices, args.options)  # 0, 1, ..., 4

        results[qid] = pred_idx
        outputs[qid] = output
        if pred_idx == answer:
            correct += 1

        acc = correct / len(results) * 100

        if args.debug or i < 3:
            print("##################################")
            print(prompt, "\n")
            print("# labeled answer:", label)
            print("# predicted answer:", prediction)
            print("# predicted index:", pred_idx)
            print("# predicted output:", output)

        if (i + 1) % args.save_every == 0 or (i + 1) == len(qids):
            print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            save_results(result_file, acc, correct, i + 1, shot_qids, args, results, outputs)
