import cv2
import numpy as np
from preprocess import preprocess_video_frames
from detect_objects import detect_frame
from embedding import object_embedding


def calculate_histogram_diff(frame1, frame2):

    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return diff

def segment_video_with_flag_frames(preprocessed_frames, threshold=0.3):    ## Use the first and the last frame as the representative for each segment
    segments = []
    start_frame_info = preprocessed_frames[0]
    for i, frame_info in enumerate(preprocessed_frames[1:], start=1):
        if i < len(preprocessed_frames) - 1:
            next_frame_info = preprocessed_frames[i + 1]       ## use histogram difference between frames to sement videos
            diff = calculate_histogram_diff(frame_info['frame'], next_frame_info['frame'])
            if diff > threshold:
                end_frame_info = preprocessed_frames[i]
                segments.append((start_frame_info, end_frame_info))
                start_frame_info = next_frame_info
    segments.append((start_frame_info, preprocessed_frames[-1]))  # the last segment
    return segments   ## return the end and start frames of each segment adn their framenum


def detect_and_generate_embeddings(flag_frame, detector_model, autoencoder_model):
    # objet detectio in key frame
    detections = detect_frame(flag_frame['frame'])
    embeddings = []
    for detection in detections:
        x, y, w, h = detection['bbox']
        cropped_img = flag_frame['frame'][y:y+h, x:x+w]
        # generate embeddings for each object in key f
        _,embedding = object_embedding(cropped_img)
        embeddings.append(embedding)
    return embeddings   ## array of object embeddings in one frame

def video_seg_embedding(video):
    """
    API to generate embedding for video segments
    """
    preprocessed_frames = preprocess_video_frames(video)
    segments = segment_video_with_flag_frames(preprocessed_frames)   ## video divided into segments
    seg_embeddings=[]
    for segment in segments:
        start_frame_info, end_frame_info = segment

        ## Embedding of the first frame

        start_embeddings = detect_and_generate_embeddings(start_frame_info)
        
        ## Embeeding of the last frame
        end_embeddings = detect_and_generate_embeddings(end_frame_info)
        
    seg_embeddings.append(start_embeddings,end_embeddings)

    return seg_embeddings