import os
import jsonlines
from tqdm import tqdm
from task import Task
import datasets
from huggingface_hub import hf_hub_download
import json
from openai import OpenAI

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
        image_contents = [str(prompt['image_content']) for prompt in prompts]
        prompt_texts = [prompt['question'] for prompt in prompts]
        answer_texts = [prompt['gt_answer'] for prompt in prompts]
        image_paths = [prompt['image_src'].split("/")[-1] for prompt in prompts]

        idxs = range(len(prompts))
        prompt_texts = [f"<IMAGE> {prompt_texts[i]}" for i in range(len(prompt_texts))]
        image_paths = [os.path.join(image_dir, image_paths[i]) for i in range(len(image_paths))]

        return idxs, prompt_texts, image_paths, image_contents, question_types, answer_texts

    def eval_task(self, subtask, model_generate_func, max_samples=None, rerun=False):
        """
        Evaluate a specific task using the provided model.
        """
        output_path = os.path.join(self.output_dir, f"predictions.jsonl")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if os.path.exists(output_path) and not rerun:
            print(f"Output file already exists: {output_path}. Skipping evaluation.")
            return
        
        idxs, prompts, image_paths, image_contents, question_types, answers = self.load_prompts()
        outputs = []

        for idx, prompt, image_path, image_content, question_type, answer in tqdm(zip(idxs, prompts, image_paths, image_contents, question_types, answers), desc=f"Processing prompts for {self.task_name}", total=len(prompts)):
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
                'image_content': image_content,
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
            image_content = output['image_content']
            answer = output['answer']
            prediction = output['prediction']
            correct = self.check_frq_answer(answer, prediction)
            score = self.check_frq_hallucination(image_content, prompt, answer, prediction)
            parsed_outputs.append({
                'idx': idx,
                'prompt': prompt,
                'image_path': image_path,
                'image_content': image_content,
                'question_type': question_type,
                'answer': answer,
                'score': score,
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
        question_type_scores = {question_type: 0 for question_type in self.question_types}
        for output in outputs:
            question_type = output['question_type']
            if question_type in question_type_counts:
                question_type_counts[question_type] += 1
                if output.get('correct', False):
                    question_type_correct[question_type] += 1
                question_type_scores[question_type] += output.get('score', 0)
            else:
                print(f"Unknown question type: {question_type}. Skipping.")
        correct_count = sum(1 for output in outputs if output.get('correct', False))
        total_score = sum(output.get('score', 0) for output in outputs)
        total_count = len(outputs)
        
        if total_count == 0:
            print(f"No outputs found. Cannot calculate accuracy.")
            return
        
        accuracy = correct_count / total_count
        avg_score = total_score / total_count
        total_count = len(outputs)

        print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})", end=' ')
        print(f"Average score: {avg_score:.4f}")
        for question_type in self.question_types:
            if question_type_counts[question_type] > 0:
                accuracy = question_type_correct[question_type] / question_type_counts[question_type]
                print(f"Accuracy for ({question_type}): {accuracy:.4f} ({question_type_correct[question_type]}/{question_type_counts[question_type]})", end=' ')
                avg_score = question_type_scores[question_type] / question_type_counts[question_type]
                print(f"Average score for ({question_type}): {avg_score:.4f}")
            else:
                print(f"No outputs found for ({question_type}). Cannot calculate accuracy.")
    
    def check_frq_hallucination(self, image_content, question, gold_answer, prediction):
        """
        Check the model's prediction for a free-response question.
        
        Args:
            gold_answer (str): The correct answer
            prediction (str): The raw text containing a free-response answer
            
        Returns:
            bool: True if the prediction matches the gold answer, False otherwise
        """
        # Configure your OpenAI API key
        client = OpenAI()
        
        # Create a prompt that asks GPT-4.1 to evaluate the free-response answer
        prompt = f"""
        Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.
        For clarity, consider these examples:
        ### Example 1:
        - Image Contents: Lynx, Cat
        - Question: How many cats can you see in this image?
        - Standard Human-Generated Answer: There are two cats visible in the image.
        - LMM Response to Evaluate: There is one cat in the image.
        - Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
        - Hallucination: Yes.
        ### Example 2:
        - Image Contents: Tree, Plant, Panda, Animal
        - Question: What can you see from this photo?
        - Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
        - LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
        - Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
        - Hallucination: No.
        ### Example 3:
        - Image Contents: Fashion accessory, Mammal, Dog, Umbrella
        - Question: Describe this image in details.
        - Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
        - LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
        - Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
        - Hallucination: Yes.
        ### Example 4:
        - Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
        - Question: Where is this photo taken?
        - Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
        - LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
        - Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
        - Hallucination: No.
        ### Example 5:
        - Image Contents: Tree, Tower
        - Question: What environmental concerns are involved in this photo?
        - Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
        - LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
        - Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
        - Hallucination: No.
        With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.
        Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.
        To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
        - Rating: 6, very informative with good analysis or reasoning, no hallucination
        - Rating: 5, very informative, no hallucination
        - Rating: 4, somewhat informative, no hallucination
        - Rating: 3, not informative, no hallucination
        - Rating: 2, very informative, with hallucination
        - Rating: 1, somewhat informative, with hallucination
        - Rating: 0, not informative, with hallucination
        Include the exact phrase "Rating: X" in your response, where X is the rating you choose.
        ### Image Contents
        {image_content}
        ### Question
        {question}
        ### Standard Human-Generated Answer
        {gold_answer}
        ### LMM Response to Evaluate
        {prediction}
        """

        prompt = prompt.format(
            image_content=image_content,
            question=question,
            gold_answer=gold_answer,
            prediction=prediction
        )
        
        # Call the GPT-4.1 nano API
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0,  # Use deterministic output for consistency
                max_tokens=1000   # We only need a short response
            )
            
            # Extract and clean the response
            answer = response.choices[0].message.content.strip().lower()
            
            print(answer)

            # Parse the YES/NO response
            for s in range(7):
                if f'rating: {s}' in answer:
                    return s
            else:
                print(f"Invalid response format from GPT-4.1 nano: {answer}")
                return False
                
        except Exception as e:
            print(f"Error calling GPT-4.1 nano API: {e}")
            return False