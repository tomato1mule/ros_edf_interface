#!/usr/bin/env python

import argparse
import rospy

from edf_env.env import UR5Env, MugEnv
from edf_env.ros_env import UR5EnvRos

def run(target_obj_pose: str, target_obj_name: str, n_distractors: int, use_support: bool):
    env = MugEnv(use_gui=True)
    env_ros = UR5EnvRos(env=env, monitor_refresh_rate=0, reset_kwargs={'mug_pose': target_obj_pose, 'mug_name': target_obj_name, 'n_distractor': n_distractors, 'use_support': use_support})
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EDF Pick-Agent')
    parser.add_argument('--target-obj-pose', type=str, default='upright',
                        help='')
    parser.add_argument('--target-obj-name', type=str, default='train/mug0',
                        help='')
    parser.add_argument('--n-distractors', type=int, default=0,
                        help='')
    parser.add_argument('--use-support', type=str, default='false',
                        help='')

    # args = parser.parse_args()
    args = parser.parse_args(rospy.myargv()[1:])
    target_obj_pose: str = args.target_obj_pose
    target_obj_name: str = args.target_obj_name
    n_distractors: int = args.n_distractors
    use_support: str = args.use_support
    if use_support == 'true' or use_support == 'True':
        use_support = True
    elif use_support == 'false' or use_support == 'False':
        use_support = False
    else:
        raise ValueError(f"Unknown value for argument --use-support: {use_support}")

    try:
        run(target_obj_pose=target_obj_pose,
            target_obj_name=target_obj_name,
            n_distractors=n_distractors,
            use_support=use_support)
    except rospy.ROSInterruptException:
        pass