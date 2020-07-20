from styx_msgs.msg import TrafficLight
import rospy
import rospkg
import os,sys
import tensorflow as tf
import numpy as np
import time
import os
import cv2

IS_DEBUG = True

class TLClassifier(object):
    # Check image source - Simulator or real world
    def __init__(self, real_world = False):                ## is_real_world = real_world
        #TODO load classifier
        self.__model_loaded = False
        self.session = None
        self.tf_graph = None
        self.prediction = None
        self.model_path = '../../../models/'
        self.load_model(real_world)


    def load_model(self, real_world):
        if real_world:
            self.model_path += 'ssd_mobilenet_v1_coco_real_dataset.pb'
        else:
            self.model_path += 'ssd_mobilenet_v1_coco_sim_dataset.pb'
        rospy.loginfo('Loading model from' + self.model_path)

        # Load graph
        self.tf_graph = load_graph(self.model_path)
        self.config = tf.ConfigProto(log_device_placement=False)

        # GPU usage
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # Timeout for inactivity
        self.config.operation_timeout_in_ms = 50000

        with self.tf_graph.as_default():
            self.tf_session = tf.Session(graph=self.tf_graph, config=self.config)

        self.__model_loaded = True
        rospy.loginfo("Loaded model successfully ")


    def get_classification(self, image, score_thresh=0.5):         ## min_score = score_thresh
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        if not self.__model_loaded:
            return TrafficLight.UNKNOWN

        tl_type = ["RED", "YELLOW", "GREEN"]                       ## tfl = tl
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        boxes, scores, classes, num = self.computation(image_np)

        # Crop image
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        for i, box in enumerate(boxes):                            ## j = i
            width = (box[3] - box[1]) * image.shape[1]
            height = (box[2] - box[0]) * image.shape[0]

            # Ignore if TL is far away
            if height < 50:  # 50px
                if IS_DEBUG:
                    rospy.loginfo("Traffic light unavailable or far away")
                return TrafficLight.UNKNOWN
            else:
                # Apply threshold
                final_scores, final_classes = get_final_results(score_thresh, scores, classes)
                if len(final_classes) == 0:
                    if IS_DEBUG:
                        rospy.loginfo("No traffic light is detected")
                    return TrafficLight.UNKNOWN

                # Within model red = 1, yellow = 2, green = 3
                # within TrafficLight messages red = 0, yellow = 1, green = 2 thus substracting 1
                if IS_DEBUG:
                    rospy.loginfo("Predicted TL color is " + tl_type[final_classes[0] - 1] + " and score is " + str(final_scores[0]))
                return final_classes[0] - 1

    def computation(self, image_np):                    ## do_computation = computation
        # Placeholders
        image_tensor = self.tf_graph.get_tensor_by_name('prefix/image_tensor:0')

        # Box represents a part where a particular object has detected
        detection_scores = self.tf_graph.get_tensor_by_name('prefix/detection_scores:0')

        # Number of predictions
        num_detections = self.tf_graph.get_tensor_by_name('prefix/num_detections:0')

        # Classification
        detection_classes = self.tf_graph.get_tensor_by_name('prefix/detection_classes:0')
        detection_boxes = self.tf_graph.get_tensor_by_name('prefix/detection_boxes:0')
        (boxes, scores, classes, num) = self.tf_session.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np})

        return boxes, scores, classes, num


def load_graph (graph_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='prefix')
    return graph


def get_final_results(score_thresh, scores, classes):
    indexes = []
    for j in range(len(classes)):          ## i = j
        if scores[j] >= score_thresh:
            indexes.append(j)


    final_scores = scores[indexes, ...]
    final_classes = classes[indexes, ...]
    return final_scores, final_classes
