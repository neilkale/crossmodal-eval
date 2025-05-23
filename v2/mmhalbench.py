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

    def get_subtasks(self, task_name):
        """
        Get the subtasks for a given task name.
        """
        if task_name == 'all':
            return self.subtasks
        elif task_name in self.subtasks:
            return [task_name]
        else:
            raise ValueError(f"Invalid task name for MMHalBench: {task_name}. Please choose from {self.subtasks} or 'all'.")

    def load_prompts(self, subtask, max_samples=None):
        """
        Load prompts and answers for the specified task.
        """
        dataset = datasets.load_dataset(
            "mmhalbench_dataset.py",
            trust_remote_code=True,
        )
        dataset = dataset['test']

        import pdb; pdb.set_trace()

        idxs, prompts, image_paths, question_types, answers = [], [], [], [], []
        i = 0
        for row in tqdm(dataset, desc=f"Loading prompts for MMHalBench {subtask}"):
            if max_samples and i >= max_samples:
                break
            idxs.append(i)
            prompts.append(row['question'])
            image_paths.append(row['image_bytes'])
            question_types.append(row['question_type'])
            answers.append(row['answer'])
            i += 1

        return idxs, prompts, image_paths, question_types, answers
    
    def eval_task(self, subtask, model_generate_func, max_samples=None, rerun=False):
        """
        Evaluate a specific task using the provided model.
        """
        output_path = os.path.join(self.output_dir, f"{subtask}.jsonl")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if os.path.exists(output_path) and not rerun:
            print(f"Output file already exists: {output_path}. Skipping evaluation.")
            return
        
        idxs, prompts, image_paths, answers = self.load_prompts(subtask)
        outputs = []

        for idx, prompt, image_path, answer in tqdm(zip(idxs, prompts, image_paths, answers), desc=f"Processing prompts for HardBLINK {subtask}", total=len(prompts)):
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

    def eval_predictions(self, subtask, max_samples=None, rerun=False):
        """
        Evaluate the predictions for a specific subtask.
        """
        output_path = os.path.join(self.output_dir, f"{subtask}.jsonl")
        
        if not os.path.exists(output_path):
            print(f"Output file does not exist: {output_path}. Skipping evaluation.")
            return
        
        with jsonlines.open(output_path, 'r') as f:
            outputs = list(f)

        if not rerun and 'parsed_prediction' in outputs[0].keys():
            print(f"Parsed predictions already exist in {output_path}. Skipping evaluation.")
            return

        parsed_outputs = []
        correct = []

        for output in tqdm(outputs, desc=f"Evaluating predictions for HardBLINK {subtask}", total=len(outputs)):
            if max_samples and len(parsed_outputs) >= max_samples:
                print(f"Reached max_samples limit: {max_samples}. Stopping evaluation.")
                break
            # try:
            idx = output['idx']
            prompt = output['prompt']
            image_path = output['image_path']
            answer = output['answer']
            prediction = output['prediction']
            parsed_prediction = self.parse_mcq_answer(prediction)
            correct = self.check_mcq_answer(answer, parsed_prediction)
            parsed_outputs.append({
                'idx': idx,
                'prompt': prompt,
                'image_path': image_path,
                'answer': answer,
                'prediction': prediction,
                'parsed_prediction': parsed_prediction,
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
        output_path = os.path.join(self.output_dir, f"{subtask}.jsonl")
        
        if not os.path.exists(output_path):
            print(f"Output file does not exist: {output_path}. Cannot calculate accuracy.")
            return
        
        with jsonlines.open(output_path, 'r') as f:
            outputs = list(f)

        correct_count = sum(1 for output in outputs if output.get('correct', False))
        total_count = len(outputs)
        
        if total_count == 0:
            print(f"No outputs found for {subtask}. Cannot calculate accuracy.")
            return
        
        accuracy = correct_count / total_count
        total_count = len(outputs)

        print(f"Accuracy for {subtask}: {accuracy:.4f} ({correct_count}/{total_count})")