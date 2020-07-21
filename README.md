# CarND Capstone Project
Self-Driving Car Engineer Nanodegree Program

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

## Individual Submission

Name 				| Email 
---------------- | ---------------------
Pushkar Mehendale | mehenpm@gmail.com

## Overview
![](/imgs/car_sim.JPG)

The purpose of this deep learning application project is to manuver a car in simulator highway while following traffic rules by using perception, path planning and control module through integration.

Here is the high level system architecture used for this project.
![](/imgs/system_architecture.JPG)

In autonomous vehicle, we work with seneors such as camera, radar, lidar that perceives the surrounding environment through perception module. Based on the perception module output different ECUs in the ego-vehicle plans the manuver through planning module and several ECUs executes through control module. In this project I work with ego-vehicles position, velocity, acceleration and image data. 

Here is a lower level diagram showing ROS nodes and topics used in this project.
![](/imgs/ros_architecture.PNG)

## Perception Module
Perception module comprises components of the detection stack, such as traffic light detection and classification.

### Traffic Light Detection Node
The tl_detector package contains a traffic light detection node `tl_detector.py` that ingests data from **/base_waypoints**, **/image_color** and **/current_pose** topics and outputs data into the **/traffic_waypoint** topic. Working with the car's current position, base waypoints and traffic light data, I found the closest traffic light position, classified the traffic light state using a pretrained SSD MobileNet V1 COCO model. To use the transfer learning skill, I referenced the object detection lesson, then learned how to work with SSD MobileNet models to train the models on traffic lights and export them as frozen graphs. The SSD MobileNet V1 COCO frozen graph was then used in the `tl_classifier.py` script to detect and classify traffic lights in the simulator. In order to increase the performance of the prediction, the images are preprocessed and scaled so that the width is max 300px.

![](/imgs/red_sim.JPG)

## Planning Module
Planning module coprises route planning and behavior planning.

### Waypoint Updater Node
The waypoint_updater package contains the waypoint updater node: `waypoint_updater.py`. This node updates the target velocity of each waypoint based on traffic light and obstacle detection data. This node gathers data from **/base_wayoints**, **/current_pose**, **/obstacle_waypoint** and **/traffic_waypoint** topics and outputs waypoints in front of car with velocities to the **/final_waypoints** topic. My development focus in the waypoint updater script was to get the closest waypoints in front of the car using KDTree, calculate the velocity and position at each waypoint while considering the deceleration calculation used on the velocity. Then I would make sure that data got publishedd to the final waypoints topic mentioned earlier. This **node publishes 20 lookahead waypoints to the final waypoints topic at 50Hz** 

~~~python
IS_DEBUG = True
LOOKAHEAD_WPS = 20 # Number of waypoints we will publish
MAX_DECEL = 2
WP_BEFORE_TRAFFICLIGHT = 2
DECEL_RATE = 1
~~~

~~~python
 # Waypoint follower recieve at 30Hz
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints_tree:
                closest_waypoint_idx = self.get_closest_waypoint_id()
                self.publish_waypoints()
~~~

## Control Module
Finally, the control subsystem is important for controlling the car's throttle, steering and brake commands.

### Drive By Wire Node
The drive-by-wire (dbw) package contains files that control the vehicle: `dbw_node.py`, `twist_controller.py`. The DBW node ingests data from **/current_velocity**, **/twist_cmd** and **/vehicle/dbw_enable** topics and processes the data to publish throttle, brake and steering commands to the **/vehicle/throttle_cmd**, **/vehicle/brake_cmd** and **/vehicle/steering_cmd** topics. This control returnes steering using the yaw_controller based on linear velocity, angular velocity and current veloity. It calculates the throttle for each stop using the PID controller and passing into that controller the velocity error and sample time.

## Simulator Communcation with ROS: Styx

The `styx` and `styx_msgs` packages are used to provide a link between the simulator and ROS. Styx is a package that contains a server for communicating with the Unity simulator and a bridge to translate and publish simulator messages to ROS topics. styx_msgs includes custom message types used in this project.

## Troubleshooting for known issues

1. If an error `-- Can't launch node, can't locate node"`occurs, adding execution privilege on those file to the account should help:
```shell
cd /home/workspace/CarND-Capstone/ros
chmod -R +x ./src
```
2. If an error `-- Could not find the required component 'dbw_mkz_msgs'. The following CMake error indicates that you either need to install the package with the same name or change your environment so that it can be found.
CMake Error at /opt/ros/kinetic/share/catkin/cmake/catkinConfig.cmake:83 (find_package):
  Could not find a package configuration file provided by "dbw_mkz_msgs" with
  any of the following names: …"` occurs, a new install of ros-kinetic-dbw-mkz-msgs will help: 
```shell
sudo apt-get update
sudo apt-get install -y ros-kinetic-dbw-mkz-msgs
cd /home/workspace/CarND-Capstone/ros
rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
```

3. There is major camera latency issue. Try connecting your laptop with LAN cable instead of WiFi. Here are open issues on GitHub about the camera latency - **[turning on the camera slows car down so auto-mode gets messed up](https://github.com/udacity/CarND-Capstone/issues/266)**.

# Udacity README

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/m-pushkar/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
