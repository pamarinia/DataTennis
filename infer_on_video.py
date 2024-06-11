import cv2
import numpy as np
import torch

from bounce_detection import BounceDetector
from player_tracker import PlayerTracker
from court_reference import CourtReference

from court_detection import CourtDetector
from ball_detection import BallDetector

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

def write_track(frames, ball_track, trace=7):
    """
    This function writes the ball track on the frames.
    Args:
        frames: A list of numpy arrays representing the frames.
        ball_track: A list of tuples. Each tuple contains the x and y coordinates of the ball.
        trace: An integer representing the number of frames to trace back.
    Returns:
        frames: A list of numpy arrays representing the frames with the ball track.
    """
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

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

def main(frames, bounces, ball_track, keypoints_track, matrix_track, player_detections, trace=8):

    final_images = []
    width_minimap = 200
    height_minimap = 400

    court_img = get_court_img()
    for i in range(len(frames)):
        frame = frames[i]
        inv_matrix = matrix_track[i]

        for j in range(trace):
            if (i-j >= 0):
                if ball_track[i-j][0]:
                    x = int(ball_track[i-j][0])
                    y = int(ball_track[i-j][1])
                    frame = cv2.circle(frame, (x, y), radius=0, color=(0, 255, 0), thickness=10-j)
                else:
                    break
        
        for kp in keypoints_track[i]:
            if kp[0] and kp[1]:
                x = int(kp[0])
                y = int(kp[1])
                frame = cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)


            #frame = cv2.warpPerspective(frame, matrix, (court_img.shape[1], court_img.shape[0]))
        final_images.append(frame)

    final_images = player_tracker.draw_bboxes(final_images, player_detections)

    

    
    minimap = cv2.resize(court_img, (width_minimap, height_minimap))

    for frame in final_images:
        height, width, _ = frame.shape
        frame[10:10+height_minimap, width - 10 -width_minimap:width-10] = minimap
    
    return final_images

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames, fps = read_video('input/Med_Djo_cutcut.mp4')

    print('Ball detection')
    ball_detector = BallDetector(model_path='models\BallTrackNet_model_best.pt', device=device)
    ball_track = ball_detector.infer_model(frames)

    print('Court detection')
    court_detector = CourtDetector(model_path='models\CourtTrackNet_model_best.pt', device=device)
    keypoints_track, matrix_track = court_detector.infer_model(frames)

    print('Player detection')
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(frames, keypoints_track[0], read_from_stub=False, stub_path='tracker_stubs/player_detection.pkl')
    
    print('Bounce detection')
    bounce_detector = BounceDetector(path_model='models\ctb_regr_bounce.cbm')
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    final_images = main(frames, bounces, ball_track, keypoints_track, matrix_track, player_detections)
    
    write_video(final_images, fps, 'outputs/Med_Djo_cut_tracked.avi')

    




