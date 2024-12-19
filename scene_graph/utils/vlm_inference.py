#!/usr/bin/env python3

from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget
import rospy
from grounded_dino_pkg.srv import grounded_dino_service_type, grounded_dino_service_typeRequest
import json
import cv2

def grounding_dino_pro(api_key, img_path, object_list):
    """Function to do API calls to gdino pro

    Args:
        api_key (string): API key for gdino pro
        img_path (string): Path to the color image
        object_list (list): list of objects to be detected

    Returns:
        list: The list of detected objects 
    """
    
    objects_to_detect = '.'.join(object_list)
    token = api_key
    while True:
        try:
            # Step 1: initialize the config        
            config = Config(token)
            
            # Step 2: initialize the client
            client = Client(config)
            
            # Step 3: run the task by DetectionTask class
            image_url = client.upload_file(img_path)       
            
            task = DetectionTask(
            image_url=image_url,
            prompts=[TextPrompt(text=objects_to_detect)],
            targets=[DetectionTarget.BBox],  # detect only bbox
            model=DetectionModel.GDino1_5_Pro,  # detect with GroundingDino-1.5-Pro model
            )
            client.run_task(task)
            break
    
        except Exception as e:
            print(e)
            print("GDINO return error.")
        
    result = task.result
    
    return result.objects # the list of detected objects

def grounding_dino(args_dict):
    """Function to call the gdino client

    Args:
        args_dict (dict): Dictionary containg the args for the function.

    Returns:
        dict: Dictionary containing bounding box coordinates, logits and phrases. 
    """
    
    model = args_dict["model"]
    color_image_path = args_dict["color_image_path"]
    color_image = cv2.imread(color_image_path)
    
    request_dict = {}
    
    if model == "SAM":
        request_dict["color_image"] = color_image.tolist()
        request_dict["model"] = model
        request_dict["bbox_list"] = args_dict["bbox_list"]
    else:
        box_threshold = args_dict["box_threshold"]
        text_threshold = args_dict["text_threshold"]
        objects_name_list = args_dict["objects_name_list"]
        request_dict["objects_name_list"] = objects_name_list
        request_dict["color_image"] = color_image.tolist()
        request_dict["box_threshold"] = box_threshold
        request_dict["text_threshold"] = text_threshold
        request_dict["model"] = model

    service_name = "/grounded_dino_service"
    rospy.wait_for_service(service_name)
    gdino_service_client = rospy.ServiceProxy(service_name, \
                                              grounded_dino_service_type)
    request = grounded_dino_service_typeRequest()  
    request = json.dumps(request_dict)
    gdino_response = gdino_service_client(request)
    gdino_response_dict = json.loads(gdino_response.response)
    
    return gdino_response_dict

def draw_bbox_gdino_pro(color_image_path, result_objects):
    """Function to create bounding boxes and add labels over objects.

    Args:
        color_image_path (string): Path to the color image.
        result_objects (_type_): Gdino pro output

    Returns:
        ndarray: Annotated color image.
    """
    
    color_image = cv2.imread(color_image_path)
    
    bbox = []
    labels = []
    scores = []
    for idx, obj in enumerate(result_objects):
        bbox.append(obj.bbox)
        labels.append(obj.category) 
        scores.append(obj.score)   
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle on the image
        cv2.rectangle(color_image, (x1,y1), (x2, y2), (0, 255, 0), 2)
        
        # Annotate the box with its label
        label = labels[i] + " " + str(round(scores[i],2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Calculate text position to ensure it doesn't get cut off
        text_x = max(x1, 0)
        text_y = max(y1 - 5, 0)
        if text_x + text_size [0] > color_image.shape[1]:
            text_x = color_image.shape[1] - text_size[0] - 5
        if text_y - text_size[1] > 0:
            text_y = y2 + text_size[1] + 5
            
        # Draw the text on the image
        cv2.putText(color_image, label, (int(text_x), int(text_y)), font, font_scale, (255, 0, 0), font_thickness)
        
    return color_image