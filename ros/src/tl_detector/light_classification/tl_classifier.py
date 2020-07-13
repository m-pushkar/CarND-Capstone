from styx_msgs.msg import TrafficLight
import rospy
import cv2
import os
import uuid

import tensorflow as tf
import numpy as np

Lights = ['Green', 'Yellow', 'Red', 'Unknown']
home = os.path.expanduser("~")

class TLClassifier(object):
    def __init__(self, site):
        #TODO load classifier
        if site == True:
            # Take model trained on real images
            path_to_model = 'light_classification/models/ssd_mobilenet_site/'
        elif site == False:
            path_to_model = 'light_classification/models/ssd_mobilenet_sim

        model = path_to_model + 'ssd_mobilenet_v1_coco_tl_2018_01_28/frozen_inference_graph.pb'

        # Load model graph
        self.detection_graph = self.load_graph(pretrained_model)

        # Extract tensors for object detection
        self.image_tensor, self.detection_boxes, self.detection_scores, self.detection_classes = self.extract_tensors()
        self.sees = tf.Session(graph = self.detection_graph)

    ## Load interface graph
    def get_graph(self, graph_file):       #load_graph = get_graph
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as carnd: ##
            serialized_graph = carnd.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
        return graph

    ## Extract tensors for object detection
    def get_tensors(self):        #extract_tensors = get_tensors
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        return image_tensor, detection_boxes, detection_scores, detection_classes

    ## Boxes with confidence level >= minimum score
    def get_boxes(self, min_score, boxes, scores, classes):     #filter_boxes = get_boxes
        index = []
        for i in range(len(classes)):
            if scores[i] >= min_score:
                index.append(i)

        filter_boxes = boxes[index,  ...]
        filter_scores = scores[index, ...]
        filter_classes = classes[index, ...]
        return filter_boxes, filter_scores, filter_classes

    ## Image coordinates conversion as original image has been normalized
    def image_coversion(self, boxes, height, width):  #to_image_coords = image_conversion
        coordinates = np.zeros_like(boxes)
        coordinates[:, 0] = boxes[:, 0] * height
        coordinates[:, 1] = boxes[:, 1] * width
        coordinates[:, 2] = boxes[:, 2] * height
        coordinates[:, 3] = boxes[:, 3] * width
        return coordinates

    ## Draw boundary boxes
    def draw_boxes(self, image, boxes, classes, scores):
        for i in range(len(boxes)):
            top, left, bot, right = boxes[i, ...]
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bot)), (255, 255, 255), 5)
            text = TRAFFIC_LIGHTS[int(classes[i])-1] + ': ' + str(int(scores[i]*100)) + '%'
            cv2.putText(image, text, (int(left - 15), int(top - 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


    def get_classification(self, image, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_np = np.expand_dims( np.asarray(img_rgb, dtype=np.uint8), 0)

        with tf.Session(graph = self.detection_graph) as sess:

            ## Detection
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                 feed_dict={self.image_tensor: image_np})

            # Crop
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            confidence_cutoff = 0.4

              # Filter with confidence_cutoff threshold
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        if img == True:
            height, width, _ = image.shape
            coordinates = self.image_coversion(boxes, height, width)
            self.draw_boxes(image, coordinates, classes, scores)
            destination = os.path.join(home, 'GitHub', 'Capstone-Program-Autonomous-Vehicle-CarND', 'docs', 'images', 'tl-detection')
            if not os.path.exists(destination):
                os.makedirs(destination)
            filename = 'traffic_light' + str(uuid.uuid4()) + '.jpg'
            file_destination = os.path.join(destination, filename)
            cv2.imwrite(file_destination, image)

        ## If lights are not detected, return unknown state
        if len(scores) <= 0:
            tl_class_id = 4
            rospy.logwarn("Traffic Light UNKNOWN")
            tl_state = TrafficLight.UNKNOWN
            rospy.logwarn("Traffic_Light_State = %s", tl_state)
            return tl_state

        ## If lights are detected, return state
        tl_class_id = int(classes[np.argmax(scores)])
        if tl_class_id == 1:
            rospy.logwarn("Traffic Light GREEN")
            tl_state = TrafficLight.GREEN
        elif tl_class_id == 2:
            rospy.logwarn("Traffic Light RED")
            tl_state = TrafficLight.RED
        elif tl_class_id == 3:
            rospy.logwarn("Traffic Light YELLOW")
            tl_state = TrafficLight.YELLOW
        rospy.logwarn("Traffic_Light_State = %s", tl_state)
        return tl_state
