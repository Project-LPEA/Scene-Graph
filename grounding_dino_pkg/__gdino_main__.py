#!/usr/bin/env python3

import sys, os
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_path)

from src import grounded_dino_client
import rospy

if __name__ == '__main__':
    gdino_client = grounded_dino_client.GdinoClient()
    gdino_client.services()
    rospy.spin()