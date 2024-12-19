#!/usr/bin/env python3

from colorama import Fore, Style
import rospy
import numpy as np
from src.img2mask import SemanticMask
import json
import cv2
import os
from grounded_dino_pkg.srv import grounded_dino_service_type, grounded_dino_service_typeResponse
import warnings


class GdinoClient():
    def __init__(self):
        """Class which gets the detection from gdino."""

        warnings.filterwarnings("ignore")
        rospy.init_node("grounded_dino_node")
        self.pkg_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
        
        # Loading the gdino model      
        print(Fore.YELLOW + "Model loading started." + Fore.RESET)
        self.semantic_mask = SemanticMask()
        print(Fore.BLUE + "GDINO & SAM are up." + Fore.RESET)

    def services(self):
        """service function to initialize the service
        """

        rospy.Service(
            "grounded_dino_service",
            grounded_dino_service_type,
            self.grounded_dino_service_callback,
        )

    def grounded_dino_service_callback(self, req):
        """Callback function to the gdino service

        Args:
            req (_type_): Dictionary containing the color image and the name of
                          objects to be detected.

        Returns:
            dict: Dictionary containing bounding box coordinates, logits, phrases
                  and the annotated image.
        """

        # Loading data from the request sent by client
        data = json.loads(req.request)        
        model = data["model"]

        # Processing the image and saving it temporarily
        color_img_array = np.array(data["color_image"])
        img_path = self.pkg_path + "grounded_dino_pkg/tmp/tmp_image.png"
        color_img_array = color_img_array[..., ::-1]  # BGR to RGB
        cv2.imwrite(img_path, color_img_array)        
        
        # DINO + SAM
        if model == "DINO_SAM":
            obj_list = data["objects_name_list"]
            obj_string = ".".join(obj_list)
            box_threshold = data["box_threshold"]
            text_threshold = data["text_threshold"]
            print(Fore.YELLOW + "Calling GDINO & SAM." + Fore.RESET)
            masks, annotated_frame_with_mask, boxes_xyxy, logits, phrases = \
                self.semantic_mask.run_dino_sam(obj_string, img_path, \
                box_threshold, text_threshold)
            print(Fore.GREEN + "Output from GDINO & SAM recieved." + Fore.RESET)
            response = grounded_dino_service_typeResponse()
            response_data_dict = {
                "bounding_box": boxes_xyxy,
                "phrases": phrases,
                "logits": logits,
                "annotated_image": list(annotated_frame_with_mask),
                "masks": masks
            }
            
        # Only DINO
        elif model == "DINO":
            obj_list = data["objects_name_list"]
            obj_string = ".".join(obj_list)
            box_threshold = data["box_threshold"]
            text_threshold = data["text_threshold"]
            print(Fore.YELLOW + "Calling GDINO" + Fore.RESET)
            annotated_frame, phrases, logits, boxes_xyxy = self.semantic_mask.run_dino(\
                obj_string, img_path, box_threshold, text_threshold)
            print(Fore.GREEN + "Output from GDINO recieved." + Fore.RESET)
            response = grounded_dino_service_typeResponse()
            response_data_dict = {
                "bounding_box": boxes_xyxy,
                "phrases": phrases,
                "logits": logits,
                "annotated_image": annotated_frame
            }
            
        # Only SAM
        elif model == "SAM":
            bbox = data["bbox_list"]
            print(Fore.YELLOW + "Calling SAM" + Fore.RESET)
            masks = self.semantic_mask.run_sam(img_path, bbox)
            print(Fore.GREEN + "Output from SAM recieved." + Fore.RESET)
            response_data_dict = {
                "masks": masks
            }
        # Sending back response to client
        response = json.dumps(response_data_dict)
        
        return response