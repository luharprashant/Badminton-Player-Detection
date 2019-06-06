import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import csv
import pandas


# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = './models/exported_graph2/frozen_inference_graph.pb'

# Path to label map file
PATH_TO_LABELS = './data/utils/label.pbtxt'

# Path to image
PATH_TO_IMAGE = './test/6annot_val2690800.jpg'

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

type = sys.argv[1]


# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
if(type == 'video'):
    cap = cv2.VideoCapture('./test/trial1.mp4')

    while(1):
        ret,frame = cap.read()
        if ret:    
            frame = cv2.resize(frame, (1080,720))
            image = frame #cv2.imread(PATH_TO_IMAGE)
            image_expanded = np.expand_dims(image, axis=0)

                # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            

            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5,
                min_score_thresh=0.92)
    
            # All the results have been drawn on image. Now display the image.
            cv2.imshow('Player detector', image)
        
        # Calc the mid point of the bottom edge of bounding box to get player position & store in array
        box_cords = []
        for i,b in enumerate(boxes[0]):
            if classes[0][i] == 1:
                if scores[0][i] >= 0.9:
                    mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                    mid_y = boxes[0][i][2]
                    row = [mid_x*1080,mid_y*720]
                    box_cords.append(row)
        # Write the array to a csv file for processing            
        wtr = csv.writer(open ('plot_box.csv', 'a'), delimiter=',', lineterminator='\n')
        for x in box_cords : 
            print (x)
            wtr.writerow ([x])

    # Press q key to close the image
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


    # Clean up
    cap.release()
    cv2.destroyAllWindows()
else:
    image = cv2.imread(PATH_TO_IMAGE)
    image = cv2.resize(image, (1080,720))
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.80)

    # Calc the mid point of the bottom edge of bounding box to get player position
    for i,b in enumerate(boxes[0]):
        if classes[0][i] == 1 and scores[0][i] >= 0.9:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = boxes[0][i][2]
            row = [mid_x,mid_y]
            print('\nX: ',mid_x,', Y: ',mid_y)
            print('\nX cord: ',mid_x*1080,', Y cord: ',mid_y*720)
    
    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)

    # Press q key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()
