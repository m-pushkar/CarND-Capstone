#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from scipy.spatial import KDTree
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

IS_DEBUG = True
LOOKAHEAD_WPS = 20 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 2
WP_BEFORE_TRAFFICLIGHT = 2
DECEL_RATE = 1
SIG_VEL = 9                    # Sigmoid velocity factor
SIG_EXP = -0.43                # Sigmoid exponent
SIG_XOFFSET = -3.72            # X-axis offset                          ## SIG_OFFSET = SIG_XOFFSET
SIG_YOFFSET = -1.512           # Y-axis offset


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_wp_idx = -1

        rospy.loginfo("Waypoint updater...")
        self.loop()

    # Waypoint follower recieve at 30Hz
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints_tree:
                closest_waypoint_idx = self.get_closest_waypoint_id()
                self.publish_waypoints()
                if IS_DEBUG:
                    rospy.loginfo("Waypoint Idx {0}".format(closest_waypoint_idx))
            else:
                if IS_DEBUG:
                    if not self.pose:
                        rospy.loginfo("Could not load pose")
                    if not self.base_waypoints:
                        rospy.loginfo("Could not load waypoints")
            rate.sleep()


    def get_closest_waypoint_id(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # Next waypoint
        closest_idx = self.waypoints_tree.query([x,y], 1)[1]

        # Check position of waypoint
        closest_coordinates = self.waypoints_2d[closest_idx]      ## closest_coord = closest_coordinates
        prev_coordinates = self.waypoints_2d[closest_idx-1]       ## prev_coord = prev_coordinates

        # Hyper plane through closest_coordinates
        cl_vect = np.array(closest_coordinates)
        prev_vect = np.array(prev_coordinates)
        pos_vect = np.array([x,y])

        # Dot-product will be +ve if vectors are in same direction
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        if IS_DEBUG:
            rospy.loginfo("Closest waypoint: [idx=%d posx=%f posy=%f]", closest_idx, closest_coordinates[0], closest_coordinates[1])

        return closest_idx

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx - WP_BEFORE_TRAFFICLIGHT, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = (SIG_VEL / (1 + np.exp(SIG_EXP * (dist + SIG_XOFFSET)))) + SIG_YOFFSET
            if vel < 1.:
                vel = 0
            if IS_DEBUG:
                rospy.loginfo("stop_idx: %d, dist: %f, vel: %f, waypoint.twist.twist.linear.x: %f]", stop_idx, dist, vel, wp.twist.twist.linear.x)
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp


    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_waypoints.header
        closest_idx = self.get_closest_waypoint_id()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def publish_waypoints(self):

        # Python do the slicing
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        if IS_DEBUG:
            rospy.loginfo('Writing base waypoints.')
        self.base_waypoints = waypoints

        # Initialize waypoints_2d first before the subscriber is initialized
        if not self.waypoints_2d:
            # Find closest waypoint to car with KDTree
            # Convert coordinates to 2d-coord
            # Add to KDTree
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]

            # Initialize tree with these 2d-waypoints [x,y]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
