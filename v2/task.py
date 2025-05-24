from openai import OpenAI
import os

class Task:
    def __init__(self, task_name, data_dir, output_dir):
        if not task_name or not data_dir or not output_dir:
            raise ValueError("task_name, data_dir, and output_dir must be specified.")
        self.task_name = task_name
        self.data_dir = data_dir
        self.output_dir = output_dir

    def parse_mcq_answer(self, prediction):
        """
        Parse the model's prediction to extract the answer using GPT-4.1 nano.
        
        Args:
            prediction (str): The raw text containing a multiple-choice question and answer
            
        Returns:
            str: The extracted answer choice (A, B, C, D, etc.)
        """
        # Configure your OpenAI API key
        client = OpenAI()
        
        # Create a prompt that asks GPT-4.1 to extract the MCQ answer
        prompt = f"""
        Extract the multiple-choice answer (A, B, C, D, etc.) from the following text.
        Only return the letter of the answer, nothing else.
        
        Text: {prediction}
        
        Answer:
        """
        
        # Call the GPT-4.1 nano API
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts multiple-choice answers from text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0,  # Use deterministic output for consistency
                max_tokens=10   # We only need a short response
            )
            
            # Extract and clean the response
            answer = response.choices[0].message.content.strip().upper()
            
            # Validate that the answer is a single letter
            if len(answer) == 1 and answer.isalpha():
                return answer
            elif len(answer) > 1:
                # If we got more than one character, try to extract just the letter
                import re
                match = re.search(r'([A-Z])', answer)
                if match:
                    return match.group(1)
            
            # If we couldn't parse a valid answer
            print(f"Invalid answer format from GPT-4.1 nano: {answer}")
            return None
            
        except Exception as e:
            print(f"Error calling GPT-4.1 nano API: {e}")
            return None

    def check_mcq_answer(self, gold_answer, parsed_prediction):
        """
        Check the model's prediction for a multiple-choice question.
        
        Args:
            gold_answer (str): The correct answer choice (A, B, C, D, etc.)
            prediction (str): The raw text containing a multiple-choice question and answer
            
        Returns:
            bool: True if the parsed prediction matches the gold answer, False otherwise
        """
        # Validate that the parsed prediction is a single letter
        if parsed_prediction and len(parsed_prediction) == 1 and parsed_prediction.isalpha():
            return parsed_prediction == gold_answer
        else:
            return False


    def check_frq_answer(self, gold_answer, prediction):
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
        Evaluate whether the given prediction matches the correct answer for a free-response question.
        Consider semantic equivalence, not just exact text matching.
        
        Correct Answer: {gold_answer}
        
        Student's Answer: {prediction}
        
        Does the student's answer match the correct answer? Answer only "YES" or "NO".
        """
        
        # Call the GPT-4.1 nano API
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that evaluates free-response answers. You should consider semantic equivalence and accept answers that convey the same meaning as the correct answer, even if worded differently."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0,  # Use deterministic output for consistency
                max_tokens=10   # We only need a short response
            )
            
            # Extract and clean the response
            answer = response.choices[0].message.content.strip().upper()
            
            # Parse the YES/NO response
            if "YES" in answer:
                return True
            elif "NO" in answer:
                return False
            else:
                # If we got an unexpected response, try to extract YES/NO
                import re
                yes_match = re.search(r'\bYES\b', answer)
                no_match = re.search(r'\bNO\b', answer)
                
                if yes_match and not no_match:
                    return True
                elif no_match and not yes_match:
                    return False
                
                # If we couldn't parse a valid response
                print(f"Invalid response format from GPT-4.1 nano: {answer}")
                return False
                
        except Exception as e:
            print(f"Error calling GPT-4.1 nano API: {e}")
            return False