import sys
import time
import threading
from typing import Optional, Tuple, List, Union, Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

import rospy
import actionlib
import tf2_ros
from ros_numpy.point_cloud2 import array_to_pointcloud2, pointcloud2_to_array
from ros_numpy.image import numpy_to_image

import moveit_commander

from sensor_msgs.msg import JointState, PointCloud2, Image, JointState
from std_msgs.msg import Header, Duration
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult, JointTolerance
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import TransformStamped, Pose, Point
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject, RobotState
# from ros_edf.srv import UpdatePointCloud, UpdatePointCloudRequest, UpdatePointCloudResponse

from edf_env.env import UR5Env
from edf_env.pc_utils import encode_pc, decode_pc
from edf_env.interface import EdfInterface
from edf_env.utils import CamData

from ros_edf.pc_utils import reconstruct_surface, mesh_o3d_to_ros






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

        self.eef_child_links =  ['robotiq_85_base_link',
                                 'left_outer_knuckle',
                                 'left_outer_finger',
                                 'left_inner_knuckle',
                                 'left_inner_finger',
                                 'right_inner_knuckle',
                                 'right_inner_finger',
                                 'right_outer_knuckle',
                                 'right_outer_finger']

    def has_eef(self) -> bool:
        return self.arm_group.has_end_effector_link()
    
    def get_eef_link_name(self) -> Optional[str]:
        if self.has_eef():
            return self.arm_group.get_end_effector_link()
        else:
            None

    def plan_pose(self, pos: np.ndarray, orn: np.ndarray,
                  versor_comes_first: bool = False,
                  start_state: Optional[RobotState] = None) -> Tuple[bool, Any, Tuple[float, int]]:
        assert pos.ndim == 1 and pos.shape[-1] == 3 and orn.ndim == 1 and orn.shape[-1] == 4 # Quaternion

        pose_goal = Pose()
        pose_goal.position.x, pose_goal.position.y, pose_goal.position.z  = pos
        if versor_comes_first:
            pose_goal.orientation.w, pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z = orn 
        else:
            pose_goal.orientation.x, pose_goal.orientation.y, pose_goal.orientation.z, pose_goal.orientation.w = orn 

        self.arm_group.clear_pose_targets()
        if start_state is not None:
            self.arm_group.set_start_state(msg=start_state)
        else:
            self.arm_group.set_start_state_to_current_state()
        self.arm_group.set_pose_target(pose_goal)

        success, plan, planning_time, error_code = self.arm_group.plan()
        plan_info: Tuple[float, int] = (planning_time, error_code)

        return success, plan, plan_info
    
    def plan_pose_checkpoints(self, positions: Iterable[np.ndarray], 
                              orns: Iterable[np.ndarray],
                              versor_comes_first: bool = False) -> Tuple[List[bool], List, List[Tuple[float, int]]]:
        assert len(positions) == len(orns)

        start_state = None
        results: List[bool] = []
        plans: List = []
        plan_infos: List[Tuple[float, int]] = []
        for pos, orn in zip(positions, orns):
            success, plan, plan_info = self.plan_pose(pos=pos, orn=orn, versor_comes_first=versor_comes_first, start_state=start_state)
            results.append(success)
            plans.append(plan)
            plan_infos.append(plan_info)

            if success:
                joint_state = JointState()
                joint_state.header = plan.joint_trajectory.header
                joint_state.header.stamp = rospy.Time.now()
                joint_state.name = plan.joint_trajectory.joint_names
                joint_state.position = plan.joint_trajectory.points[-1].positions
                joint_state.velocity = plan.joint_trajectory.points[-1].velocities
                start_state = RobotState(joint_state = joint_state)
            
            else:
                break

        return results, plans, plan_infos

    def move_to_pose(self, pos: np.ndarray, orn: np.ndarray,
                     versor_comes_first: bool = False) -> bool:
        success, plan, plan_info = self.plan_pose(pos=pos, orn=orn, versor_comes_first=versor_comes_first)
        planning_time, error_code = plan_info
        if success is True:
            result: bool = self.arm_group.execute(plan_msg=plan, wait=True)
            rospy.loginfo(f"Execution result: {result}")
            self.arm_group.stop()
            return True
        else:
            rospy.loginfo(f"Plan failed. ErrorCode: {error_code}")
            self.arm_group.stop()
            return False
        
    def follow_checkpoints(self, positions: Iterable[np.ndarray], orns: Iterable[np.ndarray], versor_comes_first: bool = False) -> bool:
        results, plans, plan_infos = self.plan_pose_checkpoints(positions=positions, orns=orns, versor_comes_first=versor_comes_first)
        plan_success = results[-1]

        results = []
        execution_success = True
        if plan_success is True:
            for i, plan in enumerate(plans):
                result: bool = self.arm_group.execute(plan_msg=plan, wait=True)
                rospy.loginfo(f"Pose_{i} Execution success: {result}")
                self.arm_group.stop()
                if not result:
                    execution_success = False
                    break

        rospy.loginfo(f"Follow Checkpoints Execution success: {execution_success}")
        return execution_success

    def control_gripper(self, gripper_val: float) -> bool:
        joint_goal = self.gripper_group.get_current_joint_values()
        joint_goal[0] = gripper_val
        self.gripper_group.clear_pose_targets()
        result = self.gripper_group.go(joint_goal, wait=True)
        self.gripper_group.stop()

        return result

    def attach_mesh(self, mesh: o3d.cuda.pybind.geometry.TriangleMesh, 
                    obj_name: str = "eef", frame: Optional[str] = None,
                    pos: np.ndarray = np.array([0.,0.,0.]), orn: np.ndarray = np.array([0., 0., 0., 1.]), versor_comes_first = False,
                    link: Optional[str] = None, touch_links: Optional[List[str]] = None):
        assert pos.ndim == orn.ndim == 1 and len(pos) == 3 and len(orn) == 4
        if link is None:
            link = self.get_eef_link_name()
            if touch_links is None:
                touch_links = [link] + self.eef_child_links
        else:
            if touch_links is None:
                touch_links = [link]
        if frame is None:
            frame = link


        ##### Create Collision Object #####
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = obj_name
        # co.header.stamp = rospy.Time.now()
        co.header.frame_id = frame
        pose_msg = Pose()
        pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = pos
        if versor_comes_first:
            pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z = orn
        else:
            pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w = orn
        co.pose = pose_msg
        co.meshes = [mesh_o3d_to_ros(mesh)]
        co.mesh_poses = [pose_msg]

        ##### Attach Collision Object #####
        aco = AttachedCollisionObject()
        aco.object = co
        aco.link_name = link
        aco.touch_links = touch_links

        ##### Submit #####
        self.scene_intf._PlanningSceneInterface__submit(aco, attach=True)

    def remove_attached_object(self, obj_name: str = "eef", link: Optional[str] = None):
        if link is None:
            link = self.get_eef_link_name()
        self.scene_intf.remove_attached_object(name=obj_name, link=link)
        time.sleep(0.1)
        self.scene_intf.remove_world_object(name=obj_name)
        time.sleep(0.1)
        self.scene_intf.remove_world_object(name=obj_name)


    # def attach_pcd(self, pcd: o3d.cuda.pybind.geometry.PointCloud, 
    #                obj_name: str, frame: Optional[str] = None,
    #                pos: np.ndarray = np.array([0.,0.,0.]), orn: np.ndarray = np.array([0., 0., 0., 1.]), versor_comes_first = False,
    #                link: Optional[str] = None, touch_links: Optional[List[str]] = None):

    



class EdfRosInterface(EdfInterface):
    def __init__(self, reference_frame: str, 
                 arm_group_name: str = "arm",
                 gripper_group_name: str = "gripper",
                 planner_id: str = "BiTRRT"):

        self.update_scene_pc_flag = False
        self.scene_pc_raw = None
        self.scene_pc = None
        self.update_eef_pc_flag = False
        self.eef_pc_raw = None
        self.eef_pc = None

        self.reference_frame = reference_frame
        self.arm_group_name = arm_group_name
        self.gripper_group_name = gripper_group_name
        self.planner_id = planner_id    
    
        rospy.init_node('edf_env_ros_interface', anonymous=True)
        self.moveit_interface = EdfMoveitInterface(pose_reference_frame=self.reference_frame, arm_group_name=self.arm_group_name, gripper_group_name=self.gripper_group_name, planner_id=planner_id, init_node=False, moveit_commander_argv=sys.argv)
        # self.request_scene_pc_update = rospy.ServiceProxy('update_scene_pointcloud', UpdatePointCloud)
        self.request_scene_pc_update = rospy.ServiceProxy('update_scene_pointcloud', Empty)
        self.scene_pc_sub = rospy.Subscriber('scene_pointcloud', PointCloud2, callback=self._scene_pc_callback)
        self.request_eef_pc_update = rospy.ServiceProxy('update_eef_pointcloud', Empty)
        self.eef_pc_sub = rospy.Subscriber('eef_pointcloud', PointCloud2, callback=self._eef_pc_callback)
        self.tf_Buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_Buffer)
        self.clear_octomap = rospy.ServiceProxy('clear_octomap', Empty)

        self.min_gripper_val = 0.0
        self.max_gripper_val = 0.725

        self.eef_link_name = self.moveit_interface.get_eef_link_name()
        if not self.eef_link_name:
            rospy.logerr("There is no end-effector!")
            raise RuntimeError("There is no end-effector!")

        self.update_scene_pc(request_update=False, timeout_sec=10.0)
        self.update_eef_pc(request_update=False, timeout_sec=10.0)

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

    def _eef_pc_callback(self, data: PointCloud2):
        if self.update_eef_pc_flag is True:
            self.eef_pc_raw = data
            self.update_eef_pc_flag = False
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
        
    def update_eef_pc(self, request_update: bool = True, timeout_sec: float = 5.0) -> bool:
        rospy.loginfo(f"Commencing end-effector point cloud update...")
        if request_update:
            self.request_eef_pc_update()
        self.update_eef_pc_flag = True


        rate = rospy.Rate(20)
        success = False
        init_time = time.time()
        while not rospy.is_shutdown():
            if self.update_eef_pc_flag is False:
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
            points, colors = decode_pc(pointcloud2_to_array(self.eef_pc_raw))
            self.eef_pc = points, colors
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

    def observe_eef(self, obs_type: str ='pointcloud', update: bool = True) -> Optional[Union[Tuple[np.ndarray, np.ndarray], List[CamData]]]:
        if obs_type == 'pointcloud':
            if update:
                update_result = self.update_eef_pc(request_update=True)
                if update_result is True:
                    return self.eef_pc
                else:
                    return False
            else:
                return self.eef_pc
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
    
    def attach(self, pcd: o3d.cuda.pybind.geometry.PointCloud):
        mesh: o3d.cuda.pybind.geometry.TriangleMesh = reconstruct_surface(pcd=pcd)
        self.moveit_interface.attach_mesh(mesh = mesh, obj_name="eef")

    def detach(self):
        self.moveit_interface.remove_attached_object(obj_name="eef")

    def move_and_observe(self, obs_type: str ='pointcloud'): 
        observe_traj = np.array([[0.0, 0.0, 1.0, 0.0, 0.00, 0.0, 0.6],
                                 [np.sqrt(1/2), 0.0, np.sqrt(1/2), 0.0, 0.00, 0.0, 0.55],
                                 [1.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.5]
                                 ])
        rospy.loginfo("Moving to observation pose")
        for observe_pose in observe_traj:
            move_results, observe_pose = self.move_to_target_pose(poses = observe_pose[None,:])
            move_result = move_results[-1]
            if not move_result:
                rospy.loginfo("Failed to arrive obseravation pose.")
                break

        if move_result:
            return self.observe_eef(obs_type=obs_type, update=True)
        else:
            return None

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