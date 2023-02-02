import sys
import time
import threading
from typing import Optional, Tuple, List, Union, Any

import numpy as np

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
    def __init__(self, init_node = False, moveit_commander_argv = sys.argv):
        moveit_commander.roscpp_initialize(moveit_commander_argv)
        if init_node:
            rospy.init_node('edf_moveit_interface', anonymous=True)

        self.robot_com = moveit_commander.RobotCommander()
        self.scene_intf = moveit_commander.PlanningSceneInterface()
        self.arm_group_name = "arm"
        self.arm_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        self.arm_group.set_planner_id("BiTRRT")
        self.arm_group.set_planning_time(0.5)
        self.arm_group.set_pose_reference_frame('map')

        self.gripper_group_name = "gripper"
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
    def __init__(self):
        rospy.init_node('edf_env_ros_interface', anonymous=True)
        self.moveit_interface = EdfMoveitInterface(init_node=False, moveit_commander_argv=sys.argv)
        # self.request_scene_pc_update = rospy.ServiceProxy('update_scene_pointcloud', UpdatePointCloud)
        self.request_scene_pc_update = rospy.ServiceProxy('update_scene_pointcloud', Empty)
        self.scene_pc_sub = rospy.Subscriber('scene_pointcloud', PointCloud2, callback=self._scene_pc_callback)

        self.min_gripper_val = 0.0
        self.max_gripper_val = 0.725


        self.update_scene_pc_flag = False
        self.scene_pc_raw = None
        self.scene_pc = None
        self.update_scene_pc(request_update=False, timeout_sec=10.0)


    def _scene_pc_callback(self, data: PointCloud2):
        if self.update_scene_pc_flag is True:
            self.scene_pc_raw = data
            self.update_scene_pc_flag = False
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
            self.scene_pc = decode_pc(pointcloud2_to_array(self.scene_pc_raw))
            rospy.loginfo(f"Scene pointcloud update success!")
            return True
        else:
            rospy.loginfo(f"Scene pointcloud update failed!")
            return False
            


    def observe_scene(self, obs_type: str ='pointcloud', update: bool = True) -> Optional[Union[Tuple[np.ndarray, np.ndarray], List[CamData]]]:
        if obs_type == 'pointcloud':
            if update:
                update_result = self.update_scene_pc(request_update=True)
                if update_result is True:
                    return self.scene_pc   # (points, colors)
                else:
                    return False
            else:
                return self.scene_pc       # (points, colors)
        elif obs_type == 'image':
            raise NotImplementedError
        else:
            raise ValueError("Wrong observation type is given.")

    def observe_ee(self, obs_type: str ='pointcloud', update: bool = True):
        raise NotImplementedError
        
    def move_to_target_pose(self, poses: np.ndarray) -> Tuple[List[bool], np.ndarray]:
        assert poses.ndim == 2 and poses.shape[-1] == 7 # [[qw,qx,qy,qz,x,y,z], ...]

        results = []
        for pose in poses:
            result_ = self.moveit_interface.move_to_pose(pos=pose[4:], orn=pose[:4], versor_comes_first=True)
            results.append(result_)
            if result_ is True:
                result_pose = pose.copy()
                break
        return results, result_pose

    def grasp(self) -> bool:
        grasp_result = self.moveit_interface.control_gripper(gripper_val=self.max_gripper_val)
        return grasp_result

    def pick(self, target_poses: np.ndarray) -> Tuple[List[bool], np.ndarray, bool, List[bool], np.ndarray]:
        assert target_poses.ndim == 2 and target_poses.shape[-1] == 7 # [[qw,qx,qy,qz,x,y,z], ...]

        pre_grasp_results, grasp_pose = self.move_to_target_pose(poses = target_poses)
        grasp_result: bool = self.grasp()
        post_grasp_poses = grasp_pose.reshape(1,7)

        N_candidate_post_grasp = 5
        post_grasp_poses = np.tile(post_grasp_poses, (N_candidate_post_grasp,1))
        post_grasp_poses[:,-1] += np.linspace(0.3, 0.1, N_candidate_post_grasp)
        post_grasp_results, final_pose = self.move_to_target_pose(poses = post_grasp_poses)
        

        return pre_grasp_results, grasp_pose, grasp_result, post_grasp_results, final_pose


    def place(self, poses):
        raise NotImplementedError