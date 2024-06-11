from tracknet import TrackNet
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import sympy
from sympy import Line
from scipy.spatial import distance
from homography import get_trans_matrix, refer_kps

class CourtDetector:
    def __init__(self, model_path=None, device=None):
        self.model = TrackNet(in_channels=3, out_channels=15)
        self.device = device
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
        self.output_width = 640
        self.output_height = 360
        self.scale = 2


    def infer_model(self, frames):
        """
        This function takes a list of frames and a model and returns the keypoints of the frames.
        Args:
            frames: A list of numpy arrays representing the frames.
            model: A PyTorch model.
        Returns:
            keypoints_track: A list of lists. Each list contains the keypoints of a frame.
        """
        keypoints_track = []
        matrix_track = []

        for frame in tqdm(frames):
            img = cv2.resize(frame, (self.output_width, self.output_height))
            img = img.astype(np.float32) / 255.0
            img = np.rollaxis(img, 2, 0)
            img = torch.tensor(img).unsqueeze(0)
            input = img.float().to(self.device)

            out = self.model(input)[0]
            preds = F.sigmoid(out).detach().cpu().numpy()
            keypoints = []
            
            for kps_num in range(15):
                heatmap = (preds[kps_num] * 255).astype(np.uint8)
                ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=10, maxRadius=25)

                if circles is not None:
                    x_pred = circles[0][0][0]*self.scale
                    y_pred = circles[0][0][1]*self.scale
                    if kps_num not in [8, 9, 12, 14]:
                        x_pred, y_pred = self.refine_kps(frame, int(x_pred), int(y_pred), crop_size=40)
                    keypoints.append((x_pred, y_pred))
                else:
                    keypoints.append(None)

            matrix_trans = get_trans_matrix(keypoints)
            keypoints = None
            if matrix_trans is not None:
                keypoints = cv2.perspectiveTransform(refer_kps, matrix_trans)
                keypoints = [np.squeeze(kp) for kp in keypoints]
                matrix_trans = cv2.invert(matrix_trans)[1]
            
            keypoints_track.append(keypoints)
            matrix_track.append(matrix_trans)

        return keypoints_track, matrix_track
    

    def postprocess(self, feature_map):
        """
        This function postprocesses the feature map to get the keypoints.
        Args:
            heatmap: A numpy array representing the feature map.    
        Returns:
            x, y: coordinates of the keypoint.
        """
        feature_map *= 255
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=10, maxRadius=25)

        x_pred, y_pred = None, None
        if circles is not None:
            x_pred = circles[0][0][0]*self.scale
            y_pred = circles[0][0][1]*self.scale
        
        return x_pred, y_pred
    

    def refine_kps(self, img, x, y, crop_size=40):
        """
        This function refines the keypoints by detecting the lines around the keypoints and finding the intersection point of the lines.
        Args:
            img: A numpy array representing the image.
            x, y : coordinates of the keypoint.
            crop_size: An integer representing the size of the crop around the keypoint.
        Returns:   
            refined_x, refined_y: refined coordinates of the keypoint.
        """
        img_height, img_width = img.shape[:2]
        x_min = max(0, x - crop_size)
        x_max = min(img_width, x + crop_size)
        y_min = max(0, y - crop_size)
        y_max = min(img_height, y + crop_size)    
        
        refined_x, refined_y = x, y
        
        img_crop = img[y_min:y_max, x_min:x_max]
        lines = self.detect_lines(img_crop)

        if len(lines) > 1:
            lines = self.merge_lines(lines)
            if len(lines) == 2:
                intersection = self.line_intersection(lines[0], lines[1])
                if intersection:
                    new_x = int(intersection[1])
                    new_y = int(intersection[0])

                    if new_x > 0 and new_x < img_crop.shape[0] and new_y > 0 and new_y < img_crop.shape[1]:
                        refined_x, refined_y = new_x, new_y
                        refined_x = x_min + new_x
                        refined_y = y_min + new_y
        return refined_x, refined_y
    

    def detect_lines(self, img):
        """
        This function detects lines in the image using HoughLinesP.
        Args:
            img: A numpy array representing the image.
        Returns:
            lines: A list of lines. Each line is a list of four integers representing the coordinates of the line.
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)[1]
        lines = cv2.HoughLinesP(img_gray, rho=1, theta=np.pi/180, threshold = 30, minLineLength=10, maxLineGap=30)
        lines = np.squeeze(lines)
        if len(lines.shape) > 0:
            if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
                lines = [lines]
        else:
            lines = []
        return lines
    

    def merge_lines(self, lines):
        """
        This function merges the lines that are close to each other.
        Args:
            lines: A list of lines. Each line is a list of four integers representing the coordinates of the line.
        Returns:
            new_lines: A list of lines. Each line is a list of four integers representing the coordinates of the line.
        """
        lines = sorted(lines, key=lambda item: item[0])
        mask = [True] * len(lines)
        new_lines = []

        for i, line in enumerate(lines):
            if mask[i]:
                for j, s_line in enumerate(lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dist1 = distance.euclidean((x1, y1), (x3, y3))
                        dist2 = distance.euclidean((x2, y2), (x4, y4))
                        if dist1 < 20 and dist2 < 20:
                            line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)],
                                            dtype=np.int32)
                            mask[i + j + 1] = False
                new_lines.append(line) 

        return new_lines      
    
    
    def line_intersection(self, line1, line2):
        """
        This function takes two lines and returns the intersection point of the lines.
        Args:
            line1, line2: A list of four integers representing the coordinates of the lines. 
        Returns:
            intersection: A list of two integers representing the intersection point of the lines.
        """
        l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
        l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

        intersection = l1.intersection(l2)
        point = None
        if len(intersection) > 0:
            if isinstance(intersection[0], sympy.geometry.point.Point2D):
                point = intersection[0].coordinates
        return point
