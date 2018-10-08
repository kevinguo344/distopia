"""
ROS Bridge
=============

Publishes the state of the system to ROS.
"""

import roslibpy
import time

class RosBridge:
    """Maintains subscribers and publishers to a ROS master via rosbridge
    """

    def __init__(self,host="localhost",port=9090):
        print("Attempting to connect to rosbridge at:{}:{}...".format(host,port) 
        self.ros = roslibpy.Ros(host=host,port=port)
        self.ros.on_ready(lambda: print('Is ROS connected?', self.ros.is_connected)
        time.sleep(10) #for now, just wait 10 seconds to see if it connects
        print("Shutting down")

if __name__ == '__main__':
    ros = RosBridge()
