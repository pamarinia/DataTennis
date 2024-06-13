import io
import os
from tkinter import Image

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging

from matplotlib import patches, pyplot as plt
from matplotlib.collections import LineCollection
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class MoveNet:
    def __init__(self): 
        module = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.model = module.signatures['serving_default']
        self.input_size = 192  
        self.MIN_CROP_KEYPOINT_SCORE = 0.1
        # Dictionary that maps from joint names to keypoint indices.
        self.KEYPOINT_DICT = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }

        # Maps bones to a matplotlib color name.
        self.KEYPOINT_EDGE_INDS_TO_COLOR = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }
    
    def infer_model(self, frames, player_detections):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        for i, frame in tqdm(enumerate(frames), total=len(frames)):
            input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for track_id, bbox in player_detections[i].items():
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Coordonnées du centre
                x, y = (x1 + x2) // 2, (y1 + y2) // 2

                # Taille de l'image à extraire
                size = 192

                # Coordonnées du coin supérieur gauche de l'image à extraire
                top_left_x = x - size // 2
                top_left_y = y - size // 2

                # Extraire l'image
                input_image = frame[top_left_y:top_left_y+size, top_left_x:top_left_x+size]
                input_image = tf.expand_dims(input_image, axis=0)
                input_image = tf.cast(input_image, dtype=tf.int32)

                results = self.model(input_image)
                keypoints_with_scores = results['output_0'].numpy()

                output_image = np.squeeze(input_image, axis=0)
                output_image = self.draw_keypoints(output_image, keypoints_with_scores)
                output_image = self.draw_connections(output_image, keypoints_with_scores)

                # Calculate padding
                original_height, original_width, _ = output_image.shape

                # Resize the output image back to the original size
                #output_image = tf.image.resize(input_image, [original_height, original_width])
                frame[top_left_y:top_left_y+size, top_left_x:top_left_x+size] = output_image
        return frames
    
    
    def draw_keypoints(self, frame, keypoints):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > self.MIN_CROP_KEYPOINT_SCORE:
                cv2.circle(frame, (int(kx), int(ky)), 3, (0,255,0), -1)
        return frame

    def draw_connections(self, frame, keypoints):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        
        for edge, color in self.KEYPOINT_EDGE_INDS_TO_COLOR.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if (c1 > self.MIN_CROP_KEYPOINT_SCORE) & (c2 > self.MIN_CROP_KEYPOINT_SCORE):      
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
        return frame