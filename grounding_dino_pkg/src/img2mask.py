#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from colorama import Fore, Style

# Grounding DINO
from GroundedSegmentAnything.GroundingDINO.groundingdino.models import build_model
from GroundedSegmentAnything.GroundingDINO.groundingdino.util import box_ops
from GroundedSegmentAnything.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundedSegmentAnything.GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundedSegmentAnything.GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from GroundedSegmentAnything.segment_anything.segment_anything import build_sam, SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt

class SemanticMask():
    def __init__(self):
        """Constructor to initialize the variables, parameters and model

        Args:
            box_threshold (float): Confidence threshold for proposed bounding box
            text_threshold (float): Confidence threshold for proposed label
        """
        
        self.pkg_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
        
        self.device = 'cuda:0'
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"  
        self.ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        self.ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"        
        self.groundingdino_model = self.load_model_hf(self.ckpt_repo_id, self.ckpt_filenmae, self.ckpt_config_filename) 
        self.sam_encoder_version = "vit_h"
        self.sam_checkpoint = self.pkg_path + 'grounded_dino_pkg/GroundedSegmentAnything/segment_anything/sam_vit_h_4b8939.pth'
        #self.sam_checkpoint = '/home/himanshu/object_centric_navigation/perception_planning/segmentation/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
        print(Fore.YELLOW + "Started loading SAM." + Fore.RESET)
        self.sam = sam_model_registry[self.sam_encoder_version](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)
        print(Fore.GREEN + "SAM loaded." + Fore.RESET)

    def load_model_hf(self,repo_id, filename, ckpt_config_filename, device='cuda:0'):
        """Function to load the necessary models.

        Args:
            repo_id (string): Name of the githubb repo id.
            filename (string): Cache file name
            ckpt_config_filename (_type_): Name of the checkpoint config file.
            device (str, optional): Which cuda to assign. Defaults to 'cuda:0'.

        Returns:
            _type_: Returns the model.
        """
        
        print(Fore.YELLOW + "Started loading GDINO." + Fore.RESET)
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file) 
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        # print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        print(Fore.GREEN + "GDINO loaded." + Fore.RESET)
        
        return model 

    def run_dino_sam(self,prompt,local_image_path, box_threshold, text_threshold):
        """Function to run both DINO and SAM.

        Args:
            prompt (string): Objects to be detected and segmented.
            local_image_path (string): Path to the image on which the detections need to happen
            box_threshold (float): Threshold for bounding box detected
            text_threshold (float): Threshold for label to each bounding box

        Returns:
            _type_: masks, annotated_frame_with_mask, boxes_xyxy, logits, phrases
        """
        
        image_source, image = load_image(local_image_path)
        boxes, logits, phrases = predict(
        model=self.groundingdino_model, 
        image=image, 
        caption=prompt, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold)
        
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        
        self.sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.device)
        masks, p , y = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        annotated_frame_with_mask = copy.deepcopy(annotated_frame)
        for mask in masks:
            annotated_frame_with_mask = self.show_mask(mask[0], annotated_frame_with_mask)
        
        return masks, annotated_frame_with_mask.tolist(), boxes_xyxy.tolist(), logits.tolist(), phrases
    
    def run_dino(self,prompt,local_image_path, box_threshold, text_threshold):
        """Function to only run DINO

        Args:
            prompt (string): Objects to be detected and segmented.
            local_image_path (string): Path to the image on which the detections need to happen
            box_threshold (float): Threshold for bounding box detected
            text_threshold (float): Threshold for label to each bounding box

        Returns:
            _type_: annotated_frame, phrases, logits, boxes_xyxy 
        """
        image_source, image = load_image(local_image_path)
        boxes, logits, phrases = predict(
        model=self.groundingdino_model, 
        image=image, 
        caption=prompt, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold)
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return annotated_frame.tolist(), phrases, logits.tolist(), boxes_xyxy.tolist()
    
    def run_sam(self, local_image_path, boxes_xyxy):
        """Function to only run SAM

        Args:
            local_image_path (string): Path to the image on which segmentation need to happen.
            boxes_xyxy (list): List of bounding box coordinates

        Returns:
            list: List of masks
        """
        
        boxes_xyxy_tensor = torch.Tensor(boxes_xyxy)
        image_source, image = load_image(local_image_path)
        self.sam_predictor.set_image(image_source)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy_tensor, image_source.shape[:2]).to(self.device)
        masks, p , y = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        return masks.tolist()