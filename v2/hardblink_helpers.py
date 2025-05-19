import os
import jsonlines
from tqdm import tqdm

def get_subtasks(task_name):
    """
    Get the subtasks for a given task name.
    """
    subtasks = ['3pointscenter', '4pointscenter', '5pointscenter']

    if task_name == 'all':
        return subtasks
    elif task_name in subtasks:
        return [task_name]
    else:
        raise ValueError(f"Invalid task name for BLINK: {task_name}. Please choose from {subtasks} or 'all'.")
    
def load_prompts(task_name, data_dir):
    prompt_path = os.path.join(data_dir, f"questions/blink_{task_name}_questions.jsonl")
    image_dir = os.path.join(data_dir, f"images/blink{task_name}")
    answer_path = os.path.join(data_dir, f"answers/blink_{task_name}_answers.jsonl")

    with jsonlines.open(prompt_path, 'r') as f, jsonlines.open(answer_path, 'r') as f2:
        prompts = list(f)
        answers = list(f2)

        idxs = [prompts[i]['question_id'] for i in range(len(prompts))]
        prompt_texts = [prompts[i]['text'] for i in range(len(prompts))]
        image_paths = [os.path.join(image_dir, prompts[i]['image']) for i in range(len(prompts))]
        answer_texts = [answers[i]['text'] for i in range(len(prompts))]

    prompt_texts = [f"<IMAGE> {prompt_texts[i]}" for i in range(len(prompt_texts))]

    return idxs, prompt_texts, image_paths, answer_texts

def eval_task(task_name, model_generate_func, data_dir, output_dir, max_samples=None, rerun=False):
    """
    Evaluate a specific task using the provided model.
    """
    output_path = os.path.join(output_dir, f"{task_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_path) and not rerun:
        print(f"Output file already exists: {output_path}. Skipping evaluation.")
        return
    
    idxs, prompts, image_paths, answers = load_prompts(task_name, data_dir)
    outputs = []

    for idx, prompt, image_path, answer in tqdm(zip(idxs, prompts, image_paths, answers), desc=f"Processing prompts for HardBLINK {task_name}", total=len(prompts)):
        if max_samples and len(outputs) >= max_samples:
            print(f"Reached max_samples limit: {max_samples}. Stopping evaluation.")
            break
        # try:
        lm_answer = model_generate_func(image_path, prompt)
        outputs.append({
            'idx': idx,
            'prompt': prompt,
            'image_path': image_path,
            'answer': answer,
            'prediction': lm_answer,
        })
        # except Exception as e:
        #     print(f"Error processing index {idx}: {e}")

    with jsonlines.open(output_path, mode='w') as f:
        f.write_all(outputs)
    print(f"Evaluation results saved to {output_path}")