""" Modules which contains the perception related functions
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from utils import transformations_util
import cv2
from copy import deepcopy

def get_color_image(topic_name):
    """Function to get the RGB image from the color image topic.

    Args:
        topic_name (string): Name of the topic on which RGB image is being published .

    Returns:
        ndarray: Image in the form of numpy array.
    """
    
    data = rospy.wait_for_message(topic_name, Image)
    bridge = CvBridge()
    color_image = bridge.imgmsg_to_cv2(data, data.encoding)
    color_image.astype(np.uint8)
    
    return color_image

def get_depth_image(topic_name):
    """FUnction to get the depth image from the depth image topic.

    Args:
        topic_name (string): Name of the topic on which the depth image is being published.

    Returns:
        ndarray: Numpy array containing depth values.
    """
    
    data = rospy.wait_for_message(topic_name, Image)
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(data, data.encoding)
    depth_image.astype(np.uint64)
    
    return depth_image

def get_camera_pose(base_frame, camera_frame):
    """Function to get the camera pose wrt base frame.

    Args:
        base_frame (_type_): Frame name of base frame.
        camera_frame (_type_): Frame name of camera frame.

    Returns:
        Pose: Pose of the camera wrt base frame. 
    """
    cam_transform2_link0 = transformations_util.get_transform_func(camera_frame, base_frame)
    #cam_transform_link0 = transformations_util.tf2_to_tf(cam_transform2_link0)
    cam_pose_link0 = transformations_util.tf_to_pose(cam_transform2_link0)   
    
    return cam_pose_link0    
    
def save_image(img, path):
    """Function to save images.

    Args:
        img (ndarray): Image stored in the form of numpy array.
        path (string): Path to where the image has to be saved.
    """
    
    cv2.imwrite(path, img)
    
def remove_edge_bbox(objectness_bbox, width, height, threshold):
    """Function to remove bounding boxes on the edge of the image.

    Args:
        objectness_bbox (list): List of the bounding boxes.
        width (int): Width of the image.
        height (int): Height of the image.
        threshold (int): How much close to the edge can the bounding box be. 

    Returns:
        _type_: _description_
    """
    objectness_bbox_copy = deepcopy(objectness_bbox)
    for i,bbox in enumerate(objectness_bbox_copy):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 < threshold or y1 < threshold or x2 > (width - threshold) \
        or y2 > (height - threshold):
            objectness_bbox.remove(bbox)
           
    return objectness_bbox

def compute_bbox_overlap(bbox_1, bbox_2):
    """Function to compute the iou between two bounding boxes

    Args:
        bbox_1 (list): Bounding box coordinates of bbox_1
        bbox_2 (list): Bounding_box coordinates of bbox_2

    Returns:
        float: Iou of between bbox_1 and bbox_2
    """
    
    # Convert to [x_min, y_min, x_max, y_max] format
    x1_min, y1_min, x1_max, y1_max = bbox_1[0], bbox_1[1], bbox_1[2], bbox_1[3]
    x2_min, y2_min, x2_max, y2_max = bbox_2[0], bbox_2[1], bbox_2[2], bbox_2[3]
    
    # Calculate intersection coordinates
    x_min_inter = max(x1_min, x2_min)
    y_min_inter = max(y1_min, y2_min)
    x_max_inter = min(x1_max, x2_max)
    y_max_inter = min(y1_max, y2_max)
    
    # Compute the width and height of the intersection rectangle
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    
    # Compute the area of the intersection rectangle
    intersection_area = inter_width * inter_height
    
    # Optionally compute the Intersection over Union (IoU)
    area_bbox1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_bbox2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area_bbox1 + area_bbox2 - intersection_area
    iou = intersection_area / union_area if union_area != 0 else 0
    
    return iou

def detected_object_structure(current_dict, object_names, object_bbox, object_score):
    
    if current_dict:
        max_key = max(map(int, current_dict.keys()))
    else:
        max_key = -1
        
    for i, (name, bbox, score) in enumerate(zip(object_names, object_bbox, object_score)):
        new_key = str(max_key+1+i)
        current_dict[new_key] = {"name": name, "bbox": bbox, "score": score}
        
    return current_dict

def find_average_3d_position(depth_image_path, mask, intrinsic_parametrs):
    
    cx, cy, fx, fy = intrinsic_parametrs
    mask = np.array(mask).reshape(720, 1280)
    mask[mask != 0] = 1
    depth_image = np.load(depth_image_path)
    mask_on_depth = mask * depth_image
    num_non_zero_elements = np.count_nonzero(mask_on_depth)
    sum_of_depths = np.sum(mask_on_depth)
    avg_depth = sum_of_depths/(num_non_zero_elements*1000)
    indices = np.argwhere(np.array(mask) == 1)
    cent_x =  indices.mean(axis=0)[1]
    cent_y =  indices.mean(axis=0)[0]
    X = (cent_x - cx) * avg_depth /fx 
    Y = (cent_y - cy) * avg_depth /fy              

    return [X, Y, avg_depth]

def get_final_output(bbox, labels, color_image_path):
    
    color_image = cv2.imread(color_image_path)
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle on the image
        cv2.rectangle(color_image, (x1,y1), (x2, y2), (0, 255, 0), 2)
        
        # Annotate the box with its label
        label = labels[i]
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