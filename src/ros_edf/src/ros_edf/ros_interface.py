import sys
import time
import threading
from typing import Optional, Tuple, List, Union, Any, Iterable, Dict
import itertools

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
from geometry_msgs.msg import TransformStamped, Pose, Point, Quaternion
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse, Trigger, TriggerRequest, TriggerResponse
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject, RobotState, RobotTrajectory
# from ros_edf.srv import UpdatePointCloud, UpdatePointCloudRequest, UpdatePointCloudResponse


import torch

from edf.data import SE3, PointCloud
from edf.env_interface import EdfInterfaceBase
from ros_edf.pc_utils import reconstruct_surface, mesh_o3d_to_ros, decode_pc



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

        self.pose_reference_frame = pose_reference_frame
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

        # TODO
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

    def get_current_pose(self, numpy: bool = False, versor_first: bool = False) -> Union[Pose, Tuple[np.ndarray]]:
        pose: Pose = self.arm_group.get_current_pose().pose

        if numpy is not True:
            return pose
        
        else:
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            if versor_first is True:
                orn = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
            else:
                orn = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            return position, orn / np.linalg.norm(orn, axis=-1, keepdims=True)
        
    def execute_plans(self, plans: Iterable[RobotTrajectory]) -> List[bool]:
        results: List[bool] = []
        for plan in plans:
            result: bool = self.arm_group.execute(plan_msg=plan, wait=True)
            results.append(result)
            self.arm_group.stop()
            if not result:
                break
        return results

    def plan_pose(self, pos: np.ndarray, orn: np.ndarray,
                  versor_comes_first: bool = False,
                  start_state: Optional[RobotState] = None) -> Tuple[bool, RobotTrajectory, RobotState, float, int]:
        assert pos.ndim == 1 and pos.shape[-1] == 3 and orn.ndim == 1 and orn.shape[-1] == 4 # Quaternion
        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        if orn.dtype != np.float64:
            orn = orn.astype(np.float64)

        orn_norm = np.linalg.norm(orn, axis=-1, keepdims=True)
        if not np.allclose(orn_norm, 1., rtol=0, atol=0.01):
            rospy.logwarn("EdfMoveitInterface.plan_pose():  Input quaternion is not normalized!")
        orn = orn / orn_norm

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

        planner_output = self.arm_group.plan()
        success: bool = planner_output[0]
        plan: RobotTrajectory = planner_output[1]
        planning_time: float = planner_output[2]
        error_code: int = planner_output[3]

        if success:
            joint_state = JointState()
            joint_state.header = plan.joint_trajectory.header
            joint_state.name = plan.joint_trajectory.joint_names
            joint_state.position = plan.joint_trajectory.points[-1].positions
            joint_state.velocity = plan.joint_trajectory.points[-1].velocities
            final_state: Optional[RobotState] = RobotState(joint_state = joint_state)
        else:
            final_state: Optional[RobotState] = None

        if success:
            rospy.loginfo(f"EDF Moveit Interface: Found a motion plan with length: {len(plan.joint_trajectory.points)}")

        return success, plan, final_state, planning_time, error_code

    def plan_waypoints_cartesian(self, positions: Iterable[np.ndarray], 
                                 orns: Iterable[np.ndarray],
                                 cartesian_step: float,                 # One JointTrajectory point for each cartesian steps (in meters)
                                 cspace_step_thr: float,                    # Allowed jump threshold in C-space
                                 avoid_collision: bool = False,
                                 versor_comes_first: bool = False,
                                 start_state: Optional[RobotState] = None,
                                 success_fraction: Optional[float] = 0.95,
                                 ) -> Tuple[RobotTrajectory, float, RobotState]:
        waypoints: List[Pose] = []
        
        self.arm_group.clear_pose_targets()
        if start_state is not None:
            self.arm_group.set_start_state(msg=start_state)
        else:
            self.arm_group.set_start_state_to_current_state()

        for pos, orn in zip(positions, orns):
            assert pos.ndim == 1 and pos.shape[-1] == 3 and orn.ndim == 1 and orn.shape[-1] == 4 # Quaternion
            if pos.dtype != np.float64:
                pos = pos.astype(np.float64)
            if orn.dtype != np.float64:
                orn = orn.astype(np.float64)
            orn_norm = np.linalg.norm(orn, axis=-1, keepdims=True)
            if not np.allclose(orn_norm, 1., rtol=0, atol=0.01):
                rospy.logwarn("EdfMoveitInterface.plan_pose():  Input quaternion is not normalized!")
            orn = orn / orn_norm

            waypoints.append(Pose(position=Point(x=pos[0],y=pos[1],z=pos[2]), orientation=Quaternion(x=orn[0+versor_comes_first], y=orn[1+versor_comes_first], z=orn[2+versor_comes_first], w=orn[(3+versor_comes_first)%4])))

        plan, fraction = self.arm_group.compute_cartesian_path(waypoints=waypoints, eef_step=cartesian_step, jump_threshold=cspace_step_thr, avoid_collisions=avoid_collision)

        joint_state = JointState()
        joint_state.header = plan.joint_trajectory.header
        joint_state.name = plan.joint_trajectory.joint_names
        joint_state.position = plan.joint_trajectory.points[-1].positions
        joint_state.velocity = plan.joint_trajectory.points[-1].velocities
        final_state: Optional[RobotState] = RobotState(joint_state = joint_state)

        if success_fraction is not None:
            fraction = fraction >= success_fraction

        if fraction:
            info_str = f"EDF Moveit Interface: Found a Cartesian plan with length: {len(plan.joint_trajectory.points)}"
            if type(fraction) is not bool:
                info_str += f" || Path fraction: {fraction}"
            rospy.loginfo(info_str)

        return fraction, plan, final_state

    def control_gripper(self, gripper_val: float) -> bool:
        joint_goal = self.gripper_group.get_current_joint_values()
        joint_goal[0] = gripper_val
        self.gripper_group.clear_pose_targets()
        result = self.gripper_group.go(joint_goal, wait=True)
        self.gripper_group.stop()

        return result
    
    def add_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                    obj_name: str, frame: Optional[str] = None,
                    pos: np.ndarray = np.array([0.,0.,0.]), orn: np.ndarray = np.array([0., 0., 0., 1.]), versor_comes_first = False):
        
        assert pos.ndim == orn.ndim == 1 and len(pos) == 3 and len(orn) == 4
        if frame == None:
            frame = self.pose_reference_frame
        orn = orn / np.linalg.norm(orn, axis=-1, keepdims=True)

        ##### Create Collision Object #####
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = obj_name
        co.header.stamp = rospy.Time.now()
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

        self.scene_intf._PlanningSceneInterface__submit(co, attach=False)
        

    def attach_mesh(self, mesh: o3d.geometry.TriangleMesh, 
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
        orn = orn / np.linalg.norm(orn, axis=-1, keepdims=True)


        ##### Create Collision Object #####
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = obj_name
        co.header.stamp = rospy.Time.now()
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

    def clear(self):
        attached = self.scene_intf.get_attached_objects()
        for obj_name, msg in attached.items(): # msg: AttachedCollisionObject
            self.scene_intf.remove_attached_object(name=obj_name, link=msg.link_name)
        time.sleep(0.1)
        self.scene_intf.remove_world_object()

    def move_to_named_target(self, name: str) -> bool:
        js = JointState()
        for k,v in self.arm_group.get_named_target_values(name).items():
            js.name.append(k)
            js.position.append(v)
        return self.arm_group.go(joints=js, wait=True)
        


class EdfRosInterface(EdfInterfaceBase):
    def __init__(self, reference_frame: str, 
                 arm_group_name: str = "arm",
                 gripper_group_name: str = "gripper",
                 planner_id: str = "BiTRRT",
                 use_env_grasp_srv: bool = False,
                 use_env_attach_srv: bool = False,
                 ):

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
        self.reset_env = rospy.ServiceProxy('reset_env', Empty)
        self.eef_pc_sub = rospy.Subscriber('eef_pointcloud', PointCloud2, callback=self._eef_pc_callback)
        self.tf_Buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_Buffer)
        self.clear_octomap = rospy.ServiceProxy('clear_octomap', Empty)

        self.min_gripper_val = 0.0 # release
        self.max_gripper_val = 0.725 # grasp (0.725 max)

        self.eef_link_name = self.moveit_interface.get_eef_link_name()
        if not self.eef_link_name:
            rospy.logerr("There is no end-effector!")
            raise RuntimeError("There is no end-effector!")
        
        if use_env_attach_srv is True:
            self.request_env_attach = rospy.ServiceProxy('env_attach_srv', Trigger)
            self.request_env_detach = rospy.ServiceProxy('env_detach_srv', Trigger)
        else:
            self.request_env_attach = None
            self.request_env_detach = None

        if use_env_grasp_srv is True:
            self.request_env_grasp = rospy.ServiceProxy('env_grasp_srv', Trigger)
            self.request_env_release = rospy.ServiceProxy('env_release_srv', Trigger)
        else:
            self.request_env_grasp = None
            self.request_env_release = None


        self.update_scene_pc(request_update=False, timeout_sec=10.0)
        self.update_eef_pc(request_update=False, timeout_sec=10.0)

    def reset(self):
        self.reset_env()
        self.moveit_interface.clear()

    def set_planning_time(self, seconds: float = 5.):
        self.moveit_interface.arm_group.set_planning_time(seconds=seconds)

    def get_frame(self, target_frame: str, source_frame: str):
        trans = self.tf_Buffer.lookup_transform(target_frame=source_frame, source_frame=target_frame, time = rospy.Time()) # transform is inverse of the frame
        pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        orn = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
        orn = orn / np.linalg.norm(orn, axis=-1, keepdims=True)

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
            orns = (inv_orn * Rotation.from_quat(orns)).as_quat()
            return inv_orn.apply(points) + inv_pos, orns / np.linalg.norm(orns, axis=-1, keepdims=True)

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

    def observe_scene(self, obs_type: str ='pointcloud', update: bool = True) -> Union[bool, PointCloud]:
        if obs_type == 'pointcloud':
            if update:
                update_result = self.update_scene_pc(request_update=True)
                if update_result is True:
                    return PointCloud.from_numpy(points=self.scene_pc[0], colors=self.scene_pc[1])
                else:
                    return False
            else:
                return PointCloud.from_numpy(points=self.scene_pc[0], colors=self.scene_pc[1])
        elif obs_type == 'image':
            raise NotImplementedError
        else:
            raise ValueError("Wrong observation type is given.")

    def observe_eef(self, obs_type: str ='pointcloud', update: bool = True) -> Union[bool, PointCloud]:
        if obs_type == 'pointcloud':
            if update:
                update_result = self.update_eef_pc(request_update=True)
                if update_result is True:
                    return PointCloud.from_numpy(points=self.eef_pc[0], colors=self.eef_pc[1])
                else:
                    return False
            else:
                return PointCloud.from_numpy(points=self.eef_pc[0], colors=self.eef_pc[1])
        elif obs_type == 'image':
            raise NotImplementedError
        else:
            raise ValueError("Wrong observation type is given.")
    
    def get_current_pose(self):
        pos, orn = self.moveit_interface.get_current_pose(numpy=True, versor_first=True)
        pose = torch.cat([torch.tensor(orn), torch.tensor(pos)], dim=0).type(torch.float32)
        poses = SE3(poses=pose.unsqueeze(-2))
        return poses

    def grasp(self) -> bool:
        if self.request_env_grasp is not None:
            result: TriggerResponse = self.request_env_grasp()
            result = result.success
        else:
            result = self.moveit_interface.control_gripper(gripper_val=self.max_gripper_val)
        return result
    
    def release(self) -> bool:
        if self.request_env_release is not None:
            result: TriggerResponse = self.request_env_release()
            result = result.success
        else:
            result = self.moveit_interface.control_gripper(gripper_val=self.min_gripper_val)
        return result
    
    def add_obj(self, pcd: PointCloud, obj_name: str):
        pcd: o3d.geometry.PointCloud = pcd.to_pcd()
        mesh: o3d.geometry.TriangleMesh = reconstruct_surface(pcd=pcd)
        self.moveit_interface.add_mesh(mesh = mesh, obj_name=obj_name)

    def attach(self, obj: PointCloud):
        obj: o3d.geometry.PointCloud = obj.to_pcd()
        obj: o3d.geometry.TriangleMesh = reconstruct_surface(pcd=obj)
        self.moveit_interface.attach_mesh(mesh = obj, obj_name="eef")
        if self.request_env_attach is not None:
            self.request_env_attach()

    def _attach_sphere(self, radius: float, pos: Iterable, color: Iterable, obj_name: str = "eef"):
        pos, color = torch.tensor(pos, dtype=torch.float64, device='cpu'), torch.tensor(color, dtype=torch.float64, device='cpu')
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.paint_uniform_color(color.detach().cpu().numpy())
        mesh_sphere.translate(pos.detach().cpu().numpy())
        self.moveit_interface.attach_mesh(mesh=mesh_sphere, obj_name=obj_name)

    def attach_placeholder(self):
        self._attach_sphere(radius=0.1, pos=[0,0,0.22], color=[0.7, 0.1, 0.1])

    def detach(self):
        self.moveit_interface.remove_attached_object(obj_name="eef")
        if self.request_env_detach is not None:
            self.request_env_detach()

    def move_plans(self, targets: Iterable[Tuple[SE3, str, Dict]], start_state: Optional[RobotState] = None) -> Tuple[List[bool], List[RobotTrajectory]]:
        plans: List[RobotTrajectory] = []
        results: List[bool] = []
        for target in targets:
            target_pose, planner_name, planner_kwargs = target

            if planner_name == 'default':
                assert len(target_pose) == 1, f"EDF ROS Interface: Only one pose should be given for each waypoint of a default planner, but {len(target_pose)} poses were given."
                planner_output = self.moveit_interface.plan_pose(pos=target_pose.points.numpy()[0], orn=target_pose.orns.numpy()[0], versor_comes_first=True, start_state=start_state)
                success: bool = planner_output[0]
                plan: RobotTrajectory = planner_output[1]
                final_state: RobotState = planner_output[2]
                
            elif planner_name == 'cartesian':
                planner_output = self.moveit_interface.plan_waypoints_cartesian(positions=target_pose.points.numpy(), orns=target_pose.orns.numpy(), versor_comes_first=True, start_state=start_state, **planner_kwargs)
                success: bool = planner_output[0]
                plan: RobotTrajectory = planner_output[1]
                final_state: RobotState = planner_output[2]

            else:
                raise ValueError(f"EDF ROS Interface: Unknown planner name ({planner_name}) is given.")

            results.append(success)
            if not success:
                break
            else:
                plans.append(plan)
                start_state = final_state

        return results, plans
    
    def execute_plans(self, plans: Iterable[RobotTrajectory]) -> List[bool]:
        results: List[bool] = self.moveit_interface.execute_plans(plans=plans)
        return results
    
    def move_to_named_target(self, name: str) -> str:
        if self.moveit_interface.move_to_named_target(name=name):
            return 'SUCCESS'
        else:
            return f"MOVE_TO_{name}_FAIL"