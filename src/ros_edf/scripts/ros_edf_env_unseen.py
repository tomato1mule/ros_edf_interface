#!/usr/bin/env python

import rospy

from edf_env.env import UR5Env, MugEnv
from edf_env.ros_env import UR5EnvRos

def run():
    env = MugEnv(use_gui=True)
    env_ros = UR5EnvRos(env=env, monitor_refresh_rate=0, reset_kwargs={'mug_pose': 'random', 'mug_name': 'random', 'n_distractor': 4, 'use_support': True})
    rospy.spin()


if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass