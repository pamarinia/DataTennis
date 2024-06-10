from moviepy.video.io.VideoFileClip import VideoFileClip

def crop_video(input_video_path, output_video_path, start_time, end_time):
    """
    This function crops a video between start_time and end_time and saves it.

    Args:
        input_video_path: A string representing the path to the input video.
        output_video_path: A string representing the path to the output video.
        start_time: A float representing the start time in seconds.
        end_time: A float representing the end time in seconds.
    """
    with VideoFileClip(input_video_path) as video:
        new_video = video.subclip(start_time, end_time)
        new_video.write_videofile(output_video_path, codec='libx264')

def reduce_video_frames(input_video_path, output_video_path, num_frames):
    """
    This function reduces the number of frames in a video and saves it.

    Args:
        input_video_path: A string representing the path to the input video.
        output_video_path: A string representing the path to the output video.
        num_frames: An integer representing the number of frames to keep.
    """
    with VideoFileClip(input_video_path) as video:
        print('ok')
        duration = video.duration
        fps = video.fps
        total_frames = duration * fps

        if num_frames > total_frames:
            raise ValueError("num_frames is greater than the total number of frames in the video.")

        new_duration = duration * num_frames / total_frames
        new_video = video.subclip(0, new_duration)
        new_video.write_videofile(output_video_path, codec='libx264')

if __name__ == '__main__':
    #crop_video('input\Med_Djo_cut.mp4', 'input\Med_Djo_cutcut.mp4', 0, 5)
    reduce_video_frames('input/Med_Djo_cut.mp4', 'input/Med_Djo_1_frame.mp4', 1)