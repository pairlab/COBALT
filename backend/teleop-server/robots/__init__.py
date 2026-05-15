from robots.interfaces.teleop_robot import TeleopRobot
from robots.interfaces.ik_robot import IKRobot
from robots.interfaces.osc_robot import OSCRobot

try:
    from robots.isaac_lab_robot import IsaacLabRobot
    from robots.robosuite_robot import RobosuiteRobot
    from robots.yam_robot import YAMRobot
except Exception as e:
    print(f"Warning: Could not import robot. Got error {e}")
