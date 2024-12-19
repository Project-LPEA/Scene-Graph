import rospy
import tf2_ros
import geometry_msgs.msg
from tf_conversions import posemath
import numpy as np
from geometry_msgs.msg import Pose
import tf

def get_transform_func(child_frame_var, parent_frame_var):
        
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)       
    rate = rospy.Rate(100)        
    while True:
        try:
            tag_transform_var = tf_buffer.lookup_transform(parent_frame_var,
                                                            child_frame_var,
                                                            rospy.Time(0))   
            tf_listener.unregister()
            break

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            rate.sleep()
            continue
        
    return tag_transform_var.transform

def tf_to_mat(tf_msg):
    '''this function converts the TF into Transformation matrix'''
    # frame is equal to [[Rotation matrix],[translation vector]]
    # frame = posemath.fromTf(tf)
    # # transformation matrix 
    # T_mat = posemath.toMatrix(frame)
    
    translation = tf_msg.translation
    rotation = tf_msg.rotation

    # Convert translation to numpy array
    trans = np.array([translation.x, translation.y, translation.z])

    # Convert rotation quaternion to rotation matrix
    quat = [rotation.x, rotation.y, rotation.z, rotation.w]
    rot_matrix = tf.transformations.quaternion_matrix(quat)

    # Create homogeneous transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rot_matrix[:3, :3]
    transformation_matrix[:3, 3] = trans
    
    return transformation_matrix

def tf_from_mat(matrix):
    '''this function converts the Transformation matrix into TF'''
    # transformation matrix to frame
    # frame = posemath.fromMatrix(mat)
    # # frame to tf
    # tf = posemath.toTf(frame)
    translation = matrix[:3, 3]
    rot_matrix = matrix[:3, :3]
    quat = tf.transformations.quaternion_from_matrix(matrix)
    transform_msg = geometry_msgs.msg.Transform()
    transform_msg.translation.x = translation[0]
    transform_msg.translation.y = translation[1]
    transform_msg.translation.z = translation[2]
    transform_msg.rotation.x = quat[0]
    transform_msg.rotation.y = quat[1]
    transform_msg.rotation.z = quat[2]
    transform_msg.rotation.w = quat[3]
    
    return transform_msg

def tf_to_pose(tf_msg):
    '''this function converts the TF into Pose'''
    # tf to Pose conversion
    # pose = geometry_msgs.msg.Pose()
    # pose.position = geometry_msgs.msg.Vector3(*tf[0])
    # pose.orientation = geometry_msgs.msg.Quaternion(*tf[1])
    pose_msg = geometry_msgs.msg.Pose()
    
    # Set position
    pose_msg.position.x = tf_msg.translation.x
    pose_msg.position.y = tf_msg.translation.y
    pose_msg.position.z = tf_msg.translation.z
    
    # Set orientation
    pose_msg.orientation.x = tf_msg.rotation.x
    pose_msg.orientation.y = tf_msg.rotation.y
    pose_msg.orientation.z = tf_msg.rotation.z
    pose_msg.orientation.w = tf_msg.rotation.w

    return pose_msg

def mat_to_pose(mat):
    '''this function converts the Transformation matrix into Pose'''
    tf_msg = tf_from_mat(mat)
    pose = tf_to_pose(tf_msg)
    return pose

def mat_from_pose(pose):
    '''this function converts the Transformation matrix into Pose'''
    tf_msg = tf_from_pose(pose)
    mat = tf_to_mat(tf_msg)
    return mat

def tf_from_pose(pose):
    '''this function converts the Pose into TF'''
    # position = list_from_Vector(pose.position)
    # orientation = list_from_Quaternion(pose.orientation)
    tf_msgs = geometry_msgs.msg.Transform()
    tf_msgs.translation.x = pose.position.x
    tf_msgs.translation.y = pose.position.y
    tf_msgs.translation.z = pose.position.z
    tf_msgs.rotation.x = pose.orientation.x
    tf_msgs.rotation.y = pose.orientation.y
    tf_msgs.rotation.z = pose.orientation.z
    tf_msgs.rotation.w = pose.orientation.w
    return tf_msgs

def tf_to_tf2(tf_transform):
    tf2_transform = geometry_msgs.msg.TransformStamped().transform
    tf2_transform.translation.x = tf_transform[0][0]
    tf2_transform.translation.y = tf_transform[0][1]
    tf2_transform.translation.z = tf_transform[0][2]
    tf2_transform.rotation.x = tf_transform[1][0]
    tf2_transform.rotation.y = tf_transform[1][1]
    tf2_transform.rotation.z = tf_transform[1][2]
    tf2_transform.rotation.w = tf_transform[1][3]

    return tf2_transform

def tf2_to_tf(tf2_transform):
    tf_transform = ((tf2_transform.translation.x, tf2_transform.translation.y, tf2_transform.translation.z), 
                    (tf2_transform.rotation.x, tf2_transform.rotation.y, tf2_transform.rotation.z, tf2_transform.rotation.w))

    return tf_transform

def list_from_Vector(vector):
    values = [vector.x, vector.y, vector.z]
    return values

def list_from_Quaternion(quat):
    values = [quat.x, quat.y, quat.z, quat.w]
    return values

def pose_wrt_robot_base(cam_pose, object_pose_wrt_camera):
    
    cam_mat = mat_from_pose(cam_pose)
    object_mat_wrt_base = np.matmul(
            cam_mat, [object_pose_wrt_camera[0], object_pose_wrt_camera[1], \
                object_pose_wrt_camera[2], 1])
    pose_list = [object_mat_wrt_base[0], object_mat_wrt_base[1], object_mat_wrt_base[2]]

    return pose_list

def pose_to_list(pose):
    
    pose_list = []
    pose_list.append(pose.position.x)
    pose_list.append(pose.position.y)
    pose_list.append(pose.position.z)
    pose_list.append(pose.orientation.x)
    pose_list.append(pose.orientation.y)
    pose_list.append(pose.orientation.z)
    pose_list.append(pose.orientation.w)
    
    return pose_list

def list_to_pose(pose_list):
    
    target_pose = Pose()
    target_pose.position.x = pose_list[0]
    target_pose.position.y = pose_list[1]
    target_pose.position.z = pose_list[2]
    target_pose.orientation.x = pose_list[3]
    target_pose.orientation.y = pose_list[4]
    target_pose.orientation.z = pose_list[5]
    target_pose.orientation.w = pose_list[6]
    
    return target_pose