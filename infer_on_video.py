import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial import distance

from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
from tracknet import TrackNet
from player_tracker import PlayerTracker

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def read_video(video_path):
    """
    This function reads a video from video_path and returns the frames and the frames per second.

    Args:
        video_path: A string representing the path to the video.
    
    Returns:
        frames: A list of numpy arrays representing the frames of the video.
        fps: An integer representing the frames per second of the video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(imgs_new, fps, path_output_video):
    """
    This function writes the frames to a video.

    Args:
        imgs_new: A list of numpy arrays representing the frames.
        fps: An integer representing the frames per second of the video.
        path_output_video: A string representing the path to the output video.
    """
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for img in imgs_new:
        out.write(img)
    out.release()


def add_circle_frame(frame, keypoints_track):
    """
    This function adds circles to the keypoints of the frames.

    Args:
        keypoints_track: A list of lists. Each list contains the keypoints of a frame.
    
    Returns:
        frames: A list of numpy arrays representing the frames with the keypoints.
    """
    for kp in keypoints_track:
        if kp[0] and kp[1]:
            x = int(kp[0])
            y = int(kp[1])
            frame = cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return frame

def infer_model(frames, model):
    """
    This function takes a list of frames and a model and returns the keypoints of the frames.
    
    Args:
        frames: A list of numpy arrays representing the frames.
        model: A PyTorch model.
    
    Returns:
        keypoints_track: A list of lists. Each list contains the keypoints of a frame.
    """
    keypoints_track = []
    for frame in tqdm(frames):
        img = cv2.resize(frame, (640, 360))
        img = img.astype(np.float32) / 255.0
        img = np.rollaxis(img, 2, 0)
        img = torch.tensor(img).unsqueeze(0)
        input = img.float().to(device)

        out = model(input)[0]
        preds = F.sigmoid(out).detach().cpu().numpy()

        keypoints = []
        for kps_num in range(14):
            heatmap = (preds[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap)
            if kps_num not in [8, 9, 12] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(frame, int(x_pred), int(y_pred))
            keypoints.append((x_pred, y_pred))
        #print(keypoints)

        matrix_trans = get_trans_matrix(keypoints)
        if matrix_trans is not None:
            keypoints = cv2.perspectiveTransform(refer_kps, matrix_trans)
            keypoints = [np.squeeze(kp) for kp in keypoints]
        
        keypoints_track.append(keypoints)

    return keypoints_track


def infer_model_ball_detection(frames, model):
    ball_track = [(None, None)]*2
    dists = [-1]*2
    for num in tqdm(range(2, len(frames))):
        img_1 = cv2.resize(frames[num-2], (640, 360))
        img_2 = cv2.resize(frames[num-1], (640, 360))
        img_3 = cv2.resize(frames[num], (640, 360))
        imgs = np.concatenate((img_3, img_2, img_1), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        imgs = np.expand_dims(imgs, axis=0)
        input = torch.from_numpy(imgs).float().to(device)

        out = model(input)
        preds = out.argmax(dim=1).detach().cpu().numpy()
        heatmap = (preds * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap[0], low_thresh=127, min_radius=2, max_radius=7)

        ball_track.append((x_pred, y_pred))
        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)
    return ball_track, dists

def remove_outliers(ball_track, dists, max_dist=100):
    """
    This function removes the outliers from the ball_track.

    Args:
        ball_track: A list of tuples. Each tuple contains the (x, y) coordinates of the ball.
        dists: A list of distances between the keypoints.
        max_dist: An integer representing the maximum distance between the keypoints.   
    
    Returns:
        ball_track: A list of tuples. Each tuple contains the (x, y) coordinates of the ball.
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track

def interpolation(coords):

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


def write_track(frames, ball_track, trace=7):
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    frame = cv2.circle(frame, (x, y), radius=0, color=(0, 255, 0), thickness=10-i)
                else:
                    break
        frames[num] = frame
    return frames

if __name__ == '__main__':

    CourtTrackNet = TrackNet(in_channels=3, out_channels=15)
    BallTrackNet = TrackNet(in_channels=9, out_channels=256)
    player_tracker = PlayerTracker(model_path='yolov8x')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BallTrackNet = BallTrackNet.to(device)
    CourtTrackNet = CourtTrackNet.to(device)

    BallTrackNet.load_state_dict(torch.load('models/BallTrackNet_model_best.pt'))
    CourtTrackNet.load_state_dict(torch.load('models/CourtTrackNet_model_best.pt'))

    BallTrackNet.eval()
    CourtTrackNet.eval()

    frames, fps = read_video('input/Med_Djo_cutcut.mp4')

    keypoints_track = infer_model(frames, CourtTrackNet)
    ball_track, dists = infer_model_ball_detection(frames, BallTrackNet)
    player_detections = player_tracker.detect_frames(frames, read_from_stub=False, stub_path='tracker_stubs/player_detection.pkl')
    
    ball_track = remove_outliers(ball_track, dists)
    ball_track = interpolation(ball_track)

    for frame, keypoints in zip(frames, keypoints_track):
        frame = add_circle_frame(frame, keypoints)
    frames = write_track(frames, ball_track, trace=7)

    frames = player_tracker.draw_bboxes(frames, player_detections)

    write_video(frames, fps, 'outputs/Med_Djo_cut_tracked.avi')




