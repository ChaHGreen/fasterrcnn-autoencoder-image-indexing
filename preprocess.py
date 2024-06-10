import cv2
from moviepy.editor import VideoFileClip
import numpy as np

def preprocess_video_frames(video_path, skip_frames=10, frame_size=(224, 224), scale=True, normalize=True):
    processed_frames = []
    timestamps = []

    # Open the video file
    video_clip = VideoFileClip(video_path)

    ## fps for the video to calculate timestamp
    fps = video_clip.fps

    # Iterate over each frame in the video
    for frame_count, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
        if frame_count % skip_frames == 0:
            # Convert BGR to RGB
            frame_rgb = frame[:, :, ::-1]

            # Resize the frame
            resized_frame = cv2.resize(frame_rgb, frame_size)

            if scale:
                resized_frame = resized_frame / 255.0 ## float32

            if normalize:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                resized_frame = (resized_frame - mean) / std

            timestamp = frame_count / fps
            
            processed_frames.append({'frame': resized_frame, 'frameNum': frame_count, 'timestamp': timestamp})

    return processed_frames

if __name__ == "__main__":
    # Example usage
    video_path = "videos/How Green Roofs Can Help Cities  NPR.mp4"
    processed_frames = preprocess_video_frames(video_path, skip_frames=5, frame_size=(224, 224), scale=True, normalize=True)
    print("Number of processed frames:", len(processed_frames))