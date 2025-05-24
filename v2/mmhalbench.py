import os
import jsonlines
from tqdm import tqdm
from task import Task
import datasets
from huggingface_hub import hf_hub_download
import json

class MMHalBench(Task):

    def __init__(self, *args, **kwargs):
        super().__init__(task_name='MMHalBench', *args, **kwargs)
        self.subtasks = ['all']
        self.question_types = ['attribute', 'adversarial', 'comparison', 'counting', 'relation', 'environment', 'holistic', 'other']

    def get_subtasks(self, task_name):
        """
        Get the subtasks for a given task name.
        """
        if task_name == 'all':
            return self.subtasks
        elif task_name in self.subtasks:
            return [task_name]
        else:
            raise ValueError(f"Invalid task name for SAT: {task_name}. Please choose from {self.subtasks} or 'all'.")

    def load_prompts(self, max_samples=None):
        """
        Load prompts and answers for the specified task.
        """
        prompt_path = os.path.join(self.data_dir, f"response_template.json")
        image_dir = os.path.join(self.data_dir, f"images")

        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
        
        question_types = [prompt['question_type'] for prompt in prompts]
        prompt_texts = [prompt['question'] for prompt in prompts]
        answer_texts = [prompt['gt_answer'] for prompt in prompts]
        image_paths = [prompt['image_src'].split("/")[-1] for prompt in prompts]

        idxs = range(len(prompts))
        prompt_texts = [f"<IMAGE> {prompt_texts[i]}" for i in range(len(prompt_texts))]
        image_paths = [os.path.join(image_dir, image_paths[i]) for i in range(len(image_paths))]

        return idxs, prompt_texts, image_paths, question_types, answer_texts

    def eval_task(self, subtask, model_generate_func, max_samples=None, rerun=False):
        """
        Evaluate a specific task using the provided model.
        """
        output_path = os.path.join(self.output_dir, f"predictions.jsonl")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if os.path.exists(output_path) and not rerun:
            print(f"Output file already exists: {output_path}. Skipping evaluation.")
            return
        
        idxs, prompts, image_paths, question_types, answers = self.load_prompts()
        outputs = []

        for idx, prompt, image_path, question_type, answer in tqdm(zip(idxs, prompts, image_paths, question_types, answers), desc=f"Processing prompts for {self.task_name}", total=len(prompts)):
            if max_samples and len(outputs) >= max_samples:
                print(f"Reached max_samples limit: {max_samples}. Stopping evaluation.")
                break
            # try:
            lm_answer = model_generate_func(image_path, prompt)
            outputs.append({
                'idx': idx,
                'prompt': prompt,
                'question_type': question_type,
                'image_path': image_path,
                'answer': answer,
                'prediction': lm_answer,
            })
            # except Exception as e:
            #     print(f"Error processing index {idx}: {e}")

        with jsonlines.open(output_path, mode='w') as f:
            f.write_all(outputs)
        print(f"Evaluation results saved to {output_path}")

    def eval_predictions(self, subtask, max_samples=None, rerun=False):
        """
        Evaluate the predictions for a specific subtask.
        """
        output_path = os.path.join(self.output_dir, "predictions.jsonl")
        
        if not os.path.exists(output_path):
            print(f"Output file does not exist: {output_path}. Skipping evaluation.")
            return
        
        with jsonlines.open(output_path, 'r') as f:
            outputs = list(f)

        if not rerun and 'correct' in outputs[0].keys():
            print(f"Evaluated answers already exist in {output_path}. Skipping evaluation.")
            return

        parsed_outputs = []
        correct = []

        for output in tqdm(outputs, desc=f"Evaluating predictions for {self.task_name}", total=len(outputs)):
            if max_samples and len(parsed_outputs) >= max_samples:
                print(f"Reached max_samples limit: {max_samples}. Stopping evaluation.")
                break
            # try:
            idx = output['idx']
            prompt = output['prompt']
            question_type = output['question_type']
            image_path = output['image_path']
            answer = output['answer']
            prediction = output['prediction']
            correct = self.check_frq_answer(answer, prediction)
            parsed_outputs.append({
                'idx': idx,
                'prompt': prompt,
                'image_path': image_path,
                'question_type': question_type,
                'answer': answer,
                'prediction': prediction,
                'correct': correct
            })
            # except Exception as e:
            #     print(f"Error processing index {idx}: {e}")

        with jsonlines.open(output_path, mode='w') as f:
            f.write_all(parsed_outputs)
        print(f"Evaluation results saved to {output_path}")

    def get_accuracy(self, subtask):
        """
        Calculate the accuracy of the predictions for a specific subtask.
        """
        output_path = os.path.join(self.output_dir, "predictions.jsonl")
        
        if not os.path.exists(output_path):
            print(f"Output file does not exist: {output_path}. Cannot calculate accuracy.")
            return
        
        with jsonlines.open(output_path, 'r') as f:
            outputs = list(f)

        question_type_counts = {question_type: 0 for question_type in self.question_types}
        question_type_correct = {question_type: 0 for question_type in self.question_types}
        for output in outputs:
            question_type = output['question_type']
            if question_type in question_type_counts:
                question_type_counts[question_type] += 1
                if output.get('correct', False):
                    question_type_correct[question_type] += 1
            else:
                print(f"Unknown question type: {question_type}. Skipping.")
        correct_count = sum(1 for output in outputs if output.get('correct', False))
        total_count = len(outputs)
        
        if total_count == 0:
            print(f"No outputs found. Cannot calculate accuracy.")
            return
        
        accuracy = correct_count / total_count
        total_count = len(outputs)

        print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        for question_type in self.question_types:
            if question_type_counts[question_type] > 0:
                accuracy = question_type_correct[question_type] / question_type_counts[question_type]
                print(f"Accuracy for ({question_type}): {accuracy:.4f} ({question_type_correct[question_type]}/{question_type_counts[question_type]})")
            else:
                print(f"No outputs found for ({question_type}). Cannot calculate accuracy.")
    
