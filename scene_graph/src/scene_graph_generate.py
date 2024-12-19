#!/usr/bin/env python3

import os, sys
src_path = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.insert(1, src_path)

import rospy
import yaml
import numpy as np
from copy import deepcopy
from colorama import Fore
import shutil
from std_msgs.msg import String
import json
from functools import partial
from scene_graph.srv import scene_graph_service_type, scene_graph_service_typeResponse
from scene_graph.utils import generic_utils, perception_utils, \
                                vlm_inference, vqa_inference, transformations_util


class SceneGraphGenerator:
    
    """Class which defines the scene graph generation pipeline. It takes as input 
    a RGB and Depth image and generates a scene graph using them."""
    
    def __init__(self):
        """Constructor class to initialize the variables and parameters
        """
        
        rospy.init_node("scene_graph_generator_node")
        self.pkg_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
        self.output_path = self.pkg_path + "output/pipeline_outputs/"
        generic_utils.remove_files(self.output_path)
        self.add_object_pub = rospy.Publisher("/add_object_info_topic", String, queue_size = 1)
        self.add_relation_pub = rospy.Publisher("/add_relation_info_topic", String, queue_size=1)
        self.load_params()
        self.services()
        print(Fore.BLUE + "Scene-Graph generator is up." + Fore.RESET)
        
    def load_params(self):
        """load the parameters from the params file to instance variables
        """
        params = {}
        with open(self.pkg_path + "config/params.yaml") as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        self.camera_color_topic = params["camera_color_topic"]
        self.camera_depth_topic = params["camera_depth_topic"]
        self.openai_key = params["openai_key"]
        self.gdino_pro_key = params["gdino_pro_key"]
        self.probable_objects = params["probable_objects"]
        self.camera_frame_name = params["camera_frame_name"]
        self.base_frame_name = params["base_frame_name"]
        self.box_threshold_object_prompt = params["box_threshold_object_prompt"]
        self.text_threshold_object_prompt = params["text_threshold_object_prompt"]
        self.box_threshold_object_list_prompt = params["box_threshold_object_list_prompt"]
        self.text_threshold_object_list_prompt = params["text_threshold_object_list_prompt"]
        self.edge_object_detection = params["edge_object_detection"]
        self.edge_threshold = params["edge_threshold"]
        self.VLM_labelled_detection = params["VLM_labelled_detector"]
        self.camera_intrinsic = params["camera_intrinsic"]
        self.default_gpt_detected_score = params["default_gpt_detected_score"]
        
    def services(self):
        """service function to initialize the service
        """

        service_name = "scene_graph_service"
        rospy.Service(service_name, scene_graph_service_type, self.generator)
        
    def generator(self, req):
        
        output_num = generic_utils.get_folder_name(self.output_path)
        save_dir_path = self.output_path + "{iter}".format(iter = output_num)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
            
        # Get the color image from a particular exploration site and save it.            
        color_image = perception_utils.get_color_image(self.camera_color_topic)
        color_image = color_image[...,::-1] # BGR to RGB
        color_image_path = save_dir_path + "/color_image.png"
        perception_utils.save_image(color_image, color_image_path)
        print(Fore.GREEN + "Color image recieved and saved" + Fore.RESET)
        
        # Get the depth image from a particular exploration site and save it.
        depth_image = perception_utils.get_depth_image(self.camera_depth_topic)
        depth_image_path = save_dir_path + "/depth_array.npy"
        np.save(depth_image_path, depth_image)
        print(Fore.GREEN + "Depth image recieved and saved." + Fore.RESET)
        
        # Get the pose of the camera wrt to the robot base
        cam_pose = perception_utils.get_camera_pose(self.base_frame_name, \
                                                    self.camera_frame_name)
        print(Fore.GREEN + "Camera pose recieved." + Fore.RESET)
        
        # Detect all the objects(unlabelled) in the image.
        args_dict = {"box_threshold": self.box_threshold_object_prompt,
                        "text_threshold": self.text_threshold_object_prompt,
                        "model": "DINO",
                        "color_image_path": color_image_path,
                        "objects_name_list": ["objects"]}
        objectness_dict = vlm_inference.grounding_dino(args_dict)
        print(Fore.GREEN + "Objectness detection from GDINO recieved." \
            + Fore.RESET)
        
        # Loading and saving the annotated image
        objectness_annotated_image = np.array(objectness_dict["annotated_image"])
        objectness_annotated_image_path = save_dir_path + \
            "/objecteness_annotated_image.png"
        perception_utils.save_image(objectness_annotated_image, \
                                    objectness_annotated_image_path)
        
        # If edge_object_detection is False, then check which bounding boxes
        # are on the edge of the image, and remove them
        objectness_bbox = objectness_dict["bounding_box"]
        if not(self.edge_object_detection):
            height, width, _ = objectness_annotated_image.shape
            objectness_bbox = perception_utils.remove_edge_bbox(objectness_bbox, \
                                        width, height, self.edge_threshold)
            
        # Getting labelled detections from grounded dino and saving the labelled image
        labelled_gdino_objects_category = []
        labelled_gdino_objects_bbox = []
        labelled_gdino_objects_score = []
        
        ## Gdino_Pro
        if self.VLM_labelled_detection == "gdino_pro":
            labelled_gdino_objects = vlm_inference.grounding_dino_pro(self.gdino_pro_key,\
                color_image_path, self.probable_objects)
            
            for idx, obj in enumerate(labelled_gdino_objects):
                if obj.score >= self.box_threshold_object_list_prompt:
                    labelled_gdino_objects_bbox.append(obj.bbox)
                    labelled_gdino_objects_score.append(obj.score)
                    labelled_gdino_objects_category.append(obj.category)
                
            labelled_gdino_color_image = vlm_inference.draw_bbox_gdino_pro(\
                                        color_image_path, labelled_gdino_objects)
            
        ## Gdino
        elif self.VLM_labelled_detection == "gdino":
            args_dict = {"box_threshold": self.box_threshold_object_list_prompt,
                        "text_threshold": self.text_threshold_object_list_prompt,
                        "model": "DINO",
                        "color_image_path": color_image_path,
                        "objects_name_list": self.probable_objects}
            labelled_gdino_object_dict = vlm_inference.grounding_dino(args_dict)
            
            for idx, bbox in labelled_gdino_object_dict["bounding_box"]:
                labelled_gdino_objects_bbox.append(bbox)
                labelled_gdino_objects_category.append(labelled_gdino_object_dict["phrases"][idx])
                labelled_gdino_objects_score.append(labelled_gdino_object_dict["logits"][idx])
            labelled_gdino_color_image = np.array(labelled_gdino_object_dict["annotated_image"])
            
        ## Converting all the object names to lowercase.
        labelled_gdino_objects_category = [s.lower() for s in labelled_gdino_objects_category]
        ## Saving the output from gdino in the output folder.
        labelled_gdino_color_image_path = save_dir_path + \
            "/labelled_gdino_color_image.png"
        perception_utils.save_image(labelled_gdino_color_image, \
                                        labelled_gdino_color_image_path)
        print(Fore.GREEN + "Objects detected by GDINO: {od}".format(od=labelled_gdino_objects_category) \
            + Fore.RESET)
        
        # Find those bounding boxes from objectness_bbox, which remained 
        # unlabelled in labelled_gdino_objects_bbox
        unlabelled_objectness_bbox = []
        for bbox_1 in objectness_bbox:
            overlap = False
            for bbox_2 in labelled_gdino_objects_bbox:
                iou = perception_utils.compute_bbox_overlap(bbox_1, bbox_2)
                if iou > 0.8:
                    overlap = True
                    break
            if not(overlap):
                unlabelled_objectness_bbox.append(bbox_1)
        visual_prompt_vqa = vqa_inference.draw_bbox_vqa_prompt(color_image_path, \
                                                unlabelled_objectness_bbox)
        visual_prompt_vqa_image_path = save_dir_path + "/visual_prompt_vqa.png"
        perception_utils.save_image(visual_prompt_vqa, \
                                    visual_prompt_vqa_image_path)
        
        # Get labels for the unlabelled_objectness_bbox
        ## Generating the prompts to be sent to the vqa
        if len(unlabelled_objectness_bbox) > 0:    
            box_labels = []
            for box_num in range(len(unlabelled_objectness_bbox)):
                box_labels.append(f'box_{box_num}')
            prompt_params = {
                "box_labels": box_labels,
                "probable_object_list": self.probable_objects,
            }
            prompt_path = self.pkg_path + "config/prompts.yaml"
            vqa_prompt_object_detection = vqa_inference.prompt_loader(\
                "vqa_prompt_object_detection_closed_set", prompt_params, prompt_path)
            VQA_output_file_path = save_dir_path + "/VQA_output.txt"
            vqa_inference.VQA_output_saver(vqa_prompt_object_detection, "QUESTION", \
                VQA_output_file_path)
            
            ## Calling the VQA
            print(Fore.YELLOW + "Calling VQA." + Fore.RESET)
            # vqa_output_raw = vqa_inference.call_vqa(vqa_prompt_object_detection, \
            #                                     visual_prompt_vqa_image_path, \
            #                                     self.openai_key)
            vqa_output_raw = vqa_inference.call_vqa_relations(vqa_prompt_object_detection, \
                                                    [color_image_path, visual_prompt_vqa_image_path], 
                                                    self.openai_key)
            print("vqa ouput raw: ", vqa_output_raw)
            prompt_params = {
                "object_detection_paragraph": vqa_output_raw,
                "template": """{"box_0": {"object_name": "name of object in box_0"}, 
                "box_1": {"object_name: name of object in box_1"}....}""",
                "probable_object_list": self.probable_objects
            }
            vqa_prompt_object_detection_parser_prompt = vqa_inference.prompt_loader(\
                "vqa_prompt_object_detection_closed_set_parser", prompt_params, prompt_path)
            vqa_detected_objects = vqa_inference.llm_parser(self.openai_key, vqa_prompt_object_detection_parser_prompt)
            print(Fore.YELLOW + "Output from VQA r: {o}".format(o=vqa_detected_objects) + Fore.RESET)
            vqa_inference.VQA_output_saver(vqa_detected_objects, "ANSWER", \
                VQA_output_file_path)
            
            # Saving all the detections in a structured format.
        detected_objects_dictionary = {}
        
        ## Objects detected by gdino / gdino_pro
        detected_objects_dictionary = perception_utils.detected_object_structure(\
            detected_objects_dictionary, labelled_gdino_objects_category, \
            labelled_gdino_objects_bbox, labelled_gdino_objects_score)
        
        ## Objects detected by vqa
        labelled_vqa_objects_category = []
        labelled_vqa_objects_bbox = []
        labelled_vqa_objects_score = []
        if len(unlabelled_objectness_bbox) > 0:
            for key, value in vqa_detected_objects.items():
                box_number = int(key.split('_')[1])
                if box_number <= len(unlabelled_objectness_bbox) - 1: 
                    labelled_vqa_objects_category.append(value["object_name"])
                    labelled_vqa_objects_bbox.append(unlabelled_objectness_bbox[box_number])
                    labelled_vqa_objects_score.append(self.default_gpt_detected_score)
            ## Converting the object names to lower case.
            labelled_vqa_objects_category = [s.lower() for s in labelled_vqa_objects_category]
            detected_objects_dictionary = perception_utils.detected_object_structure(\
                detected_objects_dictionary, labelled_vqa_objects_category, 
                labelled_vqa_objects_bbox, labelled_vqa_objects_score)
            
        # Get the 3D position of all the detected objects using the object mask.
        ## Get mask for all detected objects
        all_detected_objects_bbox = labelled_gdino_objects_bbox + \
            labelled_vqa_objects_bbox
        all_detected_objects_category = labelled_gdino_objects_category + \
            labelled_vqa_objects_category
        final_detection_output_img = perception_utils.get_final_output(\
            all_detected_objects_bbox, all_detected_objects_category, \
                color_image_path)
        perception_utils.save_image(final_detection_output_img, \
            save_dir_path + "/final_detection.png")
        if len(all_detected_objects_category) > 0:
            args_dict = {"color_image_path": color_image_path,
                            "model": "SAM",
                            "bbox_list": all_detected_objects_bbox}
            ## Calling SAM to detect masks
            all_detected_objects_masks_dict = vlm_inference.grounding_dino(args_dict)
            print(Fore.GREEN + "Object mask recieved." + Fore.RESET)
            all_detected_objects_masks = all_detected_objects_masks_dict["masks"]
            ## Adding the mask to the detected_objects_dict
            dict_copy = deepcopy(detected_objects_dictionary)
            for key, value in detected_objects_dictionary.items():
                dict_copy[key]["mask"] = all_detected_objects_masks[int(key)]
            detected_objects_dictionary = dict_copy
        
        # Getting the 3D postion for each object wrt robot base
        dict_copy = deepcopy(detected_objects_dictionary)
        for key, value in detected_objects_dictionary.items():
            pose_wrt_camera = perception_utils.find_average_3d_position(\
                depth_image_path, value["mask"], self.camera_intrinsic)
            dict_copy[key]["pose"] = transformations_util.pose_wrt_robot_base(\
                 cam_pose, pose_wrt_camera)
            #dict_copy[key]["pose"] = pose_wrt_camera
            
        detected_objects_dictionary = dict_copy
        
        # Publishing the objects info to the world model
        data = String()
        data = json.dumps(detected_objects_dictionary)
        self.add_object_pub.publish(data)
        
        print("Generating relations")
        image_paths = []            
        subdirectories = [entry for entry in os.listdir(self.output_path) if \
            os.path.isdir(os.path.join(self.output_path, entry))]
        
        for i, sub_dir_name in enumerate(subdirectories):
            image_paths.append(self.output_path + "{img_num}/color_image.png".format(img_num=i))
        
        prompt_params = {
            "probable_object_list": self.probable_objects
        }
        prompt_path = self.pkg_path + "config/prompts.yaml"
        vqa_prompt_object_relations = vqa_inference.prompt_loader(\
            "vqa_prompt_object_relation", prompt_params, prompt_path)
        print("Called vqa")
        vqa_output = vqa_inference.call_vqa_relations(vqa_prompt_object_relations, \
            image_paths, self.openai_key)
        print(vqa_output)
        prompt_params = {"object_relation_paragraph": vqa_output,
                            "template": """
                            {"object_relations":{
                                "onTop": [["object which is ontop", "object which is below"], 
                                        ["object which is ontop", "object which is below"],...],
                                "inside": [["object which is inside", "object inside which the other object is"), 
                                        ["object which is inside", "object inside which the other object is"],...],
                                "at": [["object_name", "table/rack"], ["object_name", "table/rack"],...]
                            }
                            }"""
                        }
        print("Called parser")
        parser_prompt = vqa_inference.prompt_loader(\
            "object_relation_parser_prompt", prompt_params, prompt_path)
        object_relation_dict = vqa_inference.llm_parser(self.openai_key, parser_prompt)
        
        for key in object_relation_dict['object_relations']:
        # Iterate through each list of relations for the current key
            for i in range(len(object_relation_dict['object_relations'][key])):
                # Convert each element in the sublist to lowercase
                object_relation_dict['object_relations'][key][i] = [element.lower() for element in object_relation_dict['object_relations'][key][i]]
            
        print(object_relation_dict)
        print("Print published")
        self.add_relation_pub.publish(json.dumps(object_relation_dict))
        
        response = scene_graph_service_typeResponse()
        response_dict = {}
        response = json.dumps(response_dict)
        return response
            
scene_graph_generator = SceneGraphGenerator()
rospy.spin()