import sys
import time
import threading
from typing import Optional, Tuple, List, Union, Any

import numpy as np
from scipy.spatial.transform import Rotation

import rospy
import actionlib
import tf2_ros
from ros_numpy.point_cloud2 import array_to_pointcloud2, pointcloud2_to_array
from ros_numpy.image import numpy_to_image

import moveit_commander

from sensor_msgs.msg import JointState, PointCloud2, Image
from std_msgs.msg import Header, Duration
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult, JointTolerance
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import TransformStamped, Pose
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
# from ros_edf.srv import UpdatePointCloud, UpdatePointCloudRequest, UpdatePointCloudResponse

from edf_env.env import UR5Env
from edf_env.pc_utils import encode_pc, decode_pc
from edf_env.interface import EdfInterface
from edf_env.utils import CamData



class EdfMoveitInterface():
    def __init__(self, pose_reference_frame: str, 
                 arm_group_name: str = "arm",
                 gripper_group_name: str = "gripper",
                 planner_id: str = "BiTRRT",
                 init_node: bool = False, 
                 moveit_commander_argv = sys.argv):
        moveit_commander.roscpp_initialize(moveit_commander_argv)
        if init_node:
            rospy.init_node('edf_moveit_interface', anonymous=True)

        self.robot_com = moveit_commander.RobotCommander()
        self.scene_intf = moveit_commander.PlanningSceneInterface()
        self.arm_group_name = arm_group_name
        self.arm_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        self.arm_group.set_planner_id(planner_id)
        self.arm_group.set_planning_time(0.5)
        self.arm_group.set_pose_reference_frame(pose_reference_frame)

        self.gripper_group_name = gripper_group_name
        self.gripper_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)
        self.gripper_group.set_planning_time(0.5)

    def plan_pose(self, pos: np.ndarray, orn: np.ndarray,
                  versor_comes_first: bool = False) -> Tuple[bool, Any, float, int]:
        assert pos.ndim == 1 and pos.shape[-1] == 3 and orn.ndim == 1 and orn.shape[-1] == 4 # Quaternion

        pose_goal = Pose()
        pose_goal.position.x, pose_goal.position.y, pose_goal.position.z  = pos
        if versor_comes_first:
            pose_goal.orientation.w, pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z = orn 
        else:
            pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w = orn 

        self.arm_group.clear_pose_targets()
        self.arm_group.set_pose_target(pose_goal)
        success, plan, planning_time, error_code = self.arm_group.plan()

        return success, plan, planning_time, error_code

    def move_to_pose(self, pos: np.ndarray, orn: np.ndarray,
                  versor_comes_first: bool = False) -> bool:
        success, plan, planning_time, error_code = self.plan_pose(pos=pos, orn=orn, versor_comes_first=versor_comes_first)
        if success is True:
            result: bool = self.arm_group.execute(plan_msg=plan, wait=True)
            rospy.loginfo(f"Execution result: {result}")
            self.arm_group.stop()
            return True
        else:
            rospy.loginfo(f"Plan failed. ErrorCode: {error_code}")
            self.arm_group.stop()
            return False

    def control_gripper(self, gripper_val: float) -> bool:
        joint_goal = self.gripper_group.get_current_joint_values()
        joint_goal[0] = gripper_val
        self.gripper_group.clear_pose_targets()
        result = self.gripper_group.go(joint_goal, wait=True)
        self.gripper_group.stop()

        return result




class EdfRosInterface(EdfInterface):
    def __init__(self, reference_frame: str, 
                 arm_group_name: str = "arm",
                 gripper_group_name: str = "gripper",
                 planner_id: str = "BiTRRT"):

        self.update_scene_pc_flag = False
        self.scene_pc_raw = None
        self.scene_pc = None
        self.update_ee_pc_flag = False
        self.ee_pc_raw = None
        self.ee_pc = None

        self.reference_frame = reference_frame
        self.arm_group_name = arm_group_name
        self.gripper_group_name = gripper_group_name
        self.planner_id = planner_id    
    
        rospy.init_node('edf_env_ros_interface', anonymous=True)
        self.moveit_interface = EdfMoveitInterface(pose_reference_frame=self.reference_frame, arm_group_name=self.arm_group_name, gripper_group_name=self.gripper_group_name, planner_id=planner_id, init_node=False, moveit_commander_argv=sys.argv)
        # self.request_scene_pc_update = rospy.ServiceProxy('update_scene_pointcloud', UpdatePointCloud)
        self.request_scene_pc_update = rospy.ServiceProxy('update_scene_pointcloud', Empty)
        self.scene_pc_sub = rospy.Subscriber('scene_pointcloud', PointCloud2, callback=self._scene_pc_callback)
        self.request_ee_pc_update = rospy.ServiceProxy('update_ee_pointcloud', Empty)
        self.ee_pc_sub = rospy.Subscriber('ee_pointcloud', PointCloud2, callback=self._ee_pc_callback)
        self.tf_Buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_Buffer)
        self.clear_octomap = rospy.ServiceProxy('clear_octomap', Empty)

        self.min_gripper_val = 0.0
        self.max_gripper_val = 0.725

        self.update_scene_pc(request_update=False, timeout_sec=10.0)
        self.update_ee_pc(request_update=False, timeout_sec=10.0)

    def get_frame(self, target_frame: str, source_frame: str):
        trans = self.tf_Buffer.lookup_transform(target_frame=source_frame, source_frame=target_frame, time = rospy.Time()) # transform is inverse of the frame
        pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        orn = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])

        return pos, Rotation.from_quat(orn)
    
    def transform_frame(self, points: np.ndarray, target_frame: str, source_frame: str, orns: Optional[np.ndarray] = None):
        assert points.ndim == 2 and points.shape[-1] == 3
        if orns is not None:
            assert orns.ndim == 2 and orns.shape[-1] == 3
        
        if target_frame == source_frame:
            if orns is None:
                return points
            else:
                return points, orns

        inv_pos, inv_orn = self.get_frame(target_frame=source_frame, source_frame=target_frame)
        if orns is None:
            return inv_orn.apply(points) + inv_pos
        else:
            return inv_orn.apply(points) + inv_pos, (inv_orn * Rotation.from_quat(orns)).as_quat()

    def _scene_pc_callback(self, data: PointCloud2):
        if self.update_scene_pc_flag is True:
            self.scene_pc_raw = data
            self.update_scene_pc_flag = False
        else:
            pass

    def _ee_pc_callback(self, data: PointCloud2):
        if self.update_ee_pc_flag is True:
            self.ee_pc_raw = data
            self.update_ee_pc_flag = False
        else:
            pass

    def update_scene_pc(self, request_update: bool = True, timeout_sec: float = 5.0) -> bool:
        rospy.loginfo(f"Commencing scene point cloud update...")
        if request_update:
            self.request_scene_pc_update()
        self.update_scene_pc_flag = True


        rate = rospy.Rate(20)
        success = False
        init_time = time.time()
        while not rospy.is_shutdown():
            if self.update_scene_pc_flag is False:
                success = True
                break
            
            if time.time() - init_time > timeout_sec:
                rospy.loginfo(f"Timeout: Scene pointcloud data subscription took more than {timeout_sec} seconds.")
                break
            else:
                rate.sleep()
            
        if success:
            rospy.loginfo(f"Processing received scene point cloud...")
            self.clear_octomap()
            points, colors = decode_pc(pointcloud2_to_array(self.scene_pc_raw))
            self.scene_pc = self.transform_frame(points=points, target_frame=self.reference_frame, source_frame=self.scene_pc_raw.header.frame_id), colors
            rospy.loginfo(f"Scene pointcloud update success!")
            return True
        else:
            rospy.loginfo(f"Scene pointcloud update failed!")
            return False
        
    def update_ee_pc(self, request_update: bool = True, timeout_sec: float = 5.0) -> bool:
        rospy.loginfo(f"Commencing end-effector point cloud update...")
        if request_update:
            self.request_ee_pc_update()
        self.update_ee_pc_flag = True


        rate = rospy.Rate(20)
        success = False
        init_time = time.time()
        while not rospy.is_shutdown():
            if self.update_ee_pc_flag is False:
                success = True
                break
            
            if time.time() - init_time > timeout_sec:
                rospy.loginfo(f"Timeout: End-effector pointcloud data subscription took more than {timeout_sec} seconds.")
                break
            else:
                rate.sleep()
            
        if success:
            rospy.loginfo(f"Processing received end-effector point cloud...")
            # self.clear_octomap()
            points, colors = decode_pc(pointcloud2_to_array(self.ee_pc_raw))
            self.ee_pc = points, colors
            rospy.loginfo(f"End-effector pointcloud update success!")
            return True
        else:
            rospy.loginfo(f"End-effector pointcloud update failed!")
            return False

    def observe_scene(self, obs_type: str ='pointcloud', update: bool = True) -> Optional[Union[Tuple[np.ndarray, np.ndarray], List[CamData]]]:
        if obs_type == 'pointcloud':
            if update:
                update_result = self.update_scene_pc(request_update=True)
                if update_result is True:
                    return self.scene_pc
                else:
                    return False
            else:
                return self.scene_pc
        elif obs_type == 'image':
            raise NotImplementedError
        else:
            raise ValueError("Wrong observation type is given.")

    def observe_ee(self, obs_type: str ='pointcloud', update: bool = True) -> Optional[Union[Tuple[np.ndarray, np.ndarray], List[CamData]]]:
        if obs_type == 'pointcloud':
            if update:
                update_result = self.update_ee_pc(request_update=True)
                if update_result is True:
                    return self.ee_pc
                else:
                    return False
            else:
                return self.ee_pc
        elif obs_type == 'image':
            raise NotImplementedError
        else:
            raise ValueError("Wrong observation type is given.")

    def move_to_target_pose(self, poses: np.ndarray) -> Tuple[List[bool], Optional[np.ndarray]]:
        assert poses.ndim == 2 and poses.shape[-1] == 7 # [[qw,qx,qy,qz,x,y,z], ...]

        results = []
        for pose in poses:
            result_ = self.moveit_interface.move_to_pose(pos=pose[4:], orn=pose[:4], versor_comes_first=True)
            results.append(result_)
            if result_ is True:
                result_pose = pose.copy()
                break
            else:
                result_pose = None
        return results, result_pose

    def grasp(self) -> bool:
        grasp_result = self.moveit_interface.control_gripper(gripper_val=self.max_gripper_val)
        return grasp_result
    
    def release(self) -> bool:
        grasp_result = self.moveit_interface.control_gripper(gripper_val=self.min_gripper_val)
        return grasp_result


    def pick(self, target_poses: np.ndarray) -> Tuple[List[bool], Optional[np.ndarray], bool, Optional[List[bool]], Optional[np.ndarray]]:
        assert target_poses.ndim == 2 and target_poses.shape[-1] == 7 # [[qw,qx,qy,qz,x,y,z], ...]
        if target_poses.dtype == np.float32 or target_poses.dtype == np.float16:
            target_poses = target_poses.astype(np.float64)

        release_result: bool = self.release()

        pre_grasp_results, grasp_pose = self.move_to_target_pose(poses = target_poses)
        
        if grasp_pose is None:
            post_grasp_poses = None
            grasp_result = False
            post_grasp_results = None
            final_pose = None
        else:
            grasp_result: bool = self.grasp()
            post_grasp_poses = grasp_pose.reshape(1,7)

            N_candidate_post_grasp = 5
            post_grasp_poses = np.tile(post_grasp_poses, (N_candidate_post_grasp,1))
            post_grasp_poses[:,-1] += np.linspace(0.3, 0.1, N_candidate_post_grasp)
            post_grasp_results, final_pose = self.move_to_target_pose(poses = post_grasp_poses)

        return pre_grasp_results, grasp_pose, grasp_result, post_grasp_results, final_pose


    def place(self, poses):
        raise NotImplementedError