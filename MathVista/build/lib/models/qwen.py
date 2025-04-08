"""
qwen_model.py

Class definition for wrapping Qwen2.5-VL
"""
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft.helpers import rescale_adapter_scale
from qwen_vl_utils import process_vision_info
import contextlib

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class QwenModel:
    def __init__(
        self,
        model: str,
        use_flash_attn: bool = True,
        weight_ensembling_ratio: float = 1,
    ) -> None:

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, 
            torch_dtype="auto", 
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
            device_map="cuda:0"  # TODO: auto
        )
        self.weight_ensembling_ratio = weight_ensembling_ratio

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.processor = AutoProcessor.from_pretrained(model)

        self.blank_image_path = "media/black.png"
    
    #### MAIN GENERATION FUNCTION USED BY THE MATHVISTA EVALUATION SCRIPT ####
    def get_response(self, user_prompt, decoded_image):
        prompt_text = "<IMAGES>" + user_prompt
        image = self.process_images([decoded_image])
        qwen_prompt = self.convert_prompt_to_qwen_format(prompt_text, image)
        responses = self.generate(
            qwen_prompt,
            system_prompt = None,
        )
        return response[0]

    #### HELPERS (from the ScienceQA implementation) ####

    def convert_prompt_to_qwen_format(self, prompt, images):
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

    def process_images(self, image_paths):
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

    def to_inputs(self, interleaved_text_images: List[Dict[str, Any]], system_prompt: Optional[str] = None):
        
        user_message = {"role": "user", "content": []}

        # Add the system prompt to the content if it is not None.
        if system_prompt:
            user_message["content"].append({"type": "text", "text": system_prompt})

        # Loop over each element in the interleaved_text_images list and add it to the content.
        for element in interleaved_text_images:
            user_message["content"].append(element)
        
        messages = [
            user_message,
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        return inputs, text, image_inputs, video_inputs

    def generate(
        self,
        interleaved_text_images: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        n: int = 1,
        cache: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        inputs, text, image_inputs, video_inputs = self.to_inputs(interleaved_text_images, system_prompt)

        self.update_idx_cache(cache, inputs, text, image_inputs, video_inputs)
        
        # Inference: Generation of the output
        if self.weight_ensembling_ratio != 1:
            cm = rescale_adapter_scale(self.model, self.weight_ensembling_ratio)
        else:
            cm = contextlib.nullcontext()
        with torch.no_grad(), cm:
            responses = []
            for bid in range(n):
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False, # TODO: Greedy decoding set here, potentially undo this?
                    temperature=1,  # TODO
                    top_p=0.9,  # TODO
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                    num_return_sequences=1,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                responses.append(output_text)

        return responses
    
    def to_metadata(self, inputs, text, image_inputs, video_inputs, num_prefill_tokens):
        # Image tokens are where inputs["input_ids"] == self.model.config.image_token_id
        image_token_indices = (inputs["input_ids"] == self.model.config.image_token_id).nonzero(as_tuple=True)[1]
        # print(image_token_indices)
        # print(len(image_token_indices))
        image_start_idx = image_token_indices[0].item()
        image_end_idx = image_token_indices[-1].item() + 1
        print(image_start_idx, image_end_idx)

        text_start_idx = image_end_idx

        # The end of text is specified by the last "\n\n"
        text = "\n\n".join(text.split("\n\n")[:-1])
        text = text[text.index("<|vision_end|>") + len("<|vision_end|>") :]
        if text:
            text_end_idx = text_start_idx + self.tokenizer(text, return_tensors="pt")["input_ids"].shape[-1]
        else:
            text_end_idx = text_start_idx

        print(text_start_idx, text_end_idx)

        metadata = {
            "image_start_idx": image_start_idx,
            "image_end_idx": image_end_idx,
            "text_start_idx": text_start_idx,
            "text_end_idx": text_end_idx,
        }

        return metadata
    
    def __call__(
        self,
        interleaved_text_images: List[Dict[str, Any]],
        response_prefix: Optional[str] = None,
        system_prompt: Optional[str] = None,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Any:
        inputs, text, image_inputs, video_inputs = self.to_inputs(interleaved_text_images, system_prompt)

        self.update_idx_cache(cache, inputs, text, image_inputs, video_inputs)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            print(outputs.keys())

        num_prefill_tokens = outputs["attentions"][0][0].shape[-1]
        print("attention shape", outputs["attentions"][0][0].shape)
        print("hidden_states shape", outputs["hidden_states"][0].shape)
        print("logits shape", outputs["logits"].shape)
        print("input_ids shape", inputs["input_ids"].shape)
        metadata = self.to_metadata(inputs, text, image_inputs, video_inputs, num_prefill_tokens)

        return outputs, metadata

    def update_idx_cache(self, cache, inputs, text, image_inputs, video_inputs):
        if cache is not None:
            # Image tokens are where inputs["input_ids"] == self.model.config.image_token_id
            image_token_indices = (inputs["input_ids"] == self.model.config.image_token_id).nonzero(as_tuple=True)[1]
            # print(image_token_indices)
            # print(len(image_token_indices))
            image_start_idx = image_token_indices[0].item()
            image_end_idx = image_token_indices[-1].item() + 1
            print(image_start_idx, image_end_idx)

            text_start_idx = image_end_idx

            # The end of text is specified by the last "\n\n"
            text = "\n\n".join(text.split("\n\n")[:-1])
            text = text[text.index("<|vision_end|>") + len("<|vision_end|>") :]
            if text:
                text_end_idx = text_start_idx + self.tokenizer(text, return_tensors="pt")["input_ids"].shape[-1]
            else:
                text_end_idx = text_start_idx

            print(text_start_idx, text_end_idx)

            cache["text_start_idx"] = text_start_idx
            cache["text_end_idx"] = text_end_idx
            cache["image_start_idx"] = image_start_idx
            cache["image_end_idx"] = image_end_idx
            cache["question_start_idx"] = text_end_idx
            cache["question_end_idx"] = inputs["input_ids"].shape[-1] - 2
            cache["textual_ids"] = list(range(cache["text_start_idx"], cache["text_end_idx"]))
            cache["visual_ids"] = list(range(cache["image_start_idx"], cache["image_end_idx"]))
            # print(cache)


if __name__ == "__main__":
    model_names = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ]
    for model_name in model_names:
        qwen_model = QwenModel(model=model_name)

        image_path = "media/icon.jpeg"
        responses = qwen_model.generate(
            interleaved_text_images=[
                {"type": "image", "image": Image.open(image_path)},
                {"type": "text", "text": "Describe the image in detail."},
            ],
            system_prompt="Reply in the style of Shakespeare.",
            n=10,
        )
        for i, response in enumerate(responses):
            print(f"Response {i + 1}:\n{response}")
            print("-" * 50)
        print(f"Model: {model_name} passed")