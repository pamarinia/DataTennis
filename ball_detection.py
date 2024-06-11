import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import distance
from tracknet import TrackNet


class BallDetector:
    def __init__(self, model_path=None, device=None):
        self.model = TrackNet(in_channels=9, out_channels=256)
        self.device = device
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
        self.output_width = 640
        self.output_height = 360
        self.scale = 2

    def infer_model(self, frames):
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        for num in tqdm(range(2, len(frames))):
            img_1 = cv2.resize(frames[num-2], (self.output_width, self.output_height))
            img_2 = cv2.resize(frames[num-1], (self.output_width, self.output_height))
            img_3 = cv2.resize(frames[num], (self.output_width, self.output_height))
            imgs = np.concatenate((img_3, img_2, img_1), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            imgs = np.expand_dims(imgs, axis=0)
            input = torch.from_numpy(imgs).float().to(self.device)

            out = self.model(input)
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))

        ball_track = self.interpolation(ball_track)
            
        return ball_track
    
    def postprocess(self, feature_map, prev_pred, max_dist=80):
        """
        This function postprocesses the feature map to get the keypoints.

        Args:
            feature_map: A numpy array representing the feature map.
            prev_pred: A tuple of two integers representing the previous prediction.
            max_dist: An integer representing the maximum distance between the previous prediction and the current prediction.
        
        Returns:
            x, y : ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.output_height, self.output_width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=2, maxRadius=7)

        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*self.scale
                    y_temp = circles[0][i][1]*self.scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break                
            else:
                x = circles[0][0][0]*self.scale
                y = circles[0][0][1]*self.scale
        return x, y

    def interpolation(self, coords):

        def nan_helper(y):
            # Helper to handle indices and logical indices of NaNs.
            return np.isnan(y), lambda z: z.nonzero()[0]
        
        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

        # Interpolate the missing values
        nans, xx = nan_helper(x)
        x[nans] = np.interp(xx(nans), xx(~nans), x[~nans])
        x = np.rint(x)
        nans, yy = nan_helper(y)
        y[nans] = np.interp(yy(nans), yy(~nans), y[~nans])
        y = np.rint(y)

        return list(zip(x, y))