"""Visual Language Model related functions
"""

import base64
import requests
import cv2
import os
import yaml
from openai import OpenAI
import json
import ast

def encode_image(image_path):
    """Function to encode the image in the format needed by GPT-4 model

    Args:
        image_path (str): Path to the image file

    Returns:
        base64: encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vqa(prompt, image_path, api_key):
    """Function to call the GPT-4V model 

    Args:
        prompt (str): Spatial instruction for GPT-4 model
        image_path (str): Path to the image
        api_key (str): OpenAI API key

    Returns:
        str: Output from GPT-4 model
    """
    # Getting the base64 string
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 2000
    }
    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", \
                                headers=headers, json=payload)
            response_dict = response.json()["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print("GPT return error.")        
        
    return response_dict

def call_vqa_relations(prompt, image_paths, api_key):
    
    content_list = [{
        "type": "text",
        "text": prompt}]
    
    for i in image_paths:
        base64_image = encode_image(i)
        dict = {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                },
            }
        content_list.append(dict)
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": content_list
        }
        ],
        "max_tokens": 2000
    }
    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", \
                                headers=headers, json=payload)
            # vqa_detected_objects = ast.literal_eval(response_dict)
            break
        except Exception as e:
            print("GPT return error.")
        
    return response.json()["choices"][0]["message"]["content"]    

def draw_bbox_vqa_prompt(color_image_path, bbox):
    """Function to create bounding boxes over the unlabelled objects.

    Args:
        color_image_path (list): Path to the color image.
        bbox (list): list of all unlabelled bounding boxes
    """
    
    color_image = cv2.imread(color_image_path)  
    
    for i, box in enumerate(bbox):
            
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Draw the rectangle on the image
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Annotate the box with its label
        label = f'box_{i}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # Calculate text position to ensure it doesn't get cut off
        text_x = max(x1, 0)
        text_y = max(y1 - 5, 0)
        if text_x + text_size[0] > color_image.shape[1]:
            text_x = color_image.shape[1] - text_size[0] - 5
        if text_y - text_size[1] < 0:
            text_y = y2 + text_size[1] + 5

        # Draw the text on the image
        cv2.putText(color_image, label, (int(text_x), int(text_y)), font, font_scale, (255, 0, 0), font_thickness)

    return color_image

def prompt_loader(pipeline_stage, params, prompt_path):
    """Function to load the prompt from a yaml file

    Args:
        pipeline_stage (str): Stage of the pipeline.
        params (dict): Parameters to construct the prompt.
        prompt_path (str): Path to the prompt file.

    Returns:
        str: prompt for the LLM/VLM
    """
                
    with open(prompt_path) as stream:
        try:
            prompts_all = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    prompt_format = prompts_all[pipeline_stage]
    locals().update(params)
    prompts = eval(prompt_format)
    return prompts

def VQA_output_saver(prompt, q_or_a, file_path):
    """Function to save the VQA prompts and output.

    Args:
        prompt (_type_): Textual prompt.
        q_or_a (_type_): Prompt or VQA output flag.
        file_path (_type_): Path to the file to save in.
    """
    
    VQA_output_file = open(file_path, 'a')
    VQA_output_file.write(q_or_a + ":\n\n")
    VQA_output_file.write(str(prompt) + "\n\n")
    
def llm_parser(api_key, prompt):
    
    client = OpenAI(api_key = api_key)
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_json = json.loads((response.choices[0].message.content))
            break
        except:
            print("GPT return error.")
    return response_json

