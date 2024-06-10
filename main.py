from preprocess import preprocess_video_frames
from autoencoder import ConvAutoencoder
import psycopg2
from embedding import object_embedding
from PIL import Image
import torch
import cv2
import numpy as np
import ast

db_config = {
    'dbname': 'assignment',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'db',
    'port': '5432'
}

def search_in_database(embedding, top_k=10, threshold=0):
    db_conn = psycopg2.connect(**db_config)
    cur = db_conn.cursor()
    
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    # 使用参数化查询来提高安全性
    query = """
    SELECT * FROM detected_objects 
    WHERE score > %s
    ORDER BY vector <-> %s::vector 
    LIMIT %s;
    """
    
    cur.execute(query, (threshold, embedding_str, top_k))
    results = cur.fetchall()
    
    cur.close()
    db_conn.close()
    
    return results

def draw_bbox(frame_array, bbox_info):
    x, y, width, height = bbox_info
    cv2.rectangle(frame_array, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)
    # cv2.imshow("Frame", frame_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return frame_array


def get_frame_by_num(frameNum,video_frames_info):
    for frame_info in video_frames_info:
        if frame_info['frameNum'] == frameNum:
            return frame_info['frame']
    return None

if __name__ == "__main__":

    input_image = Image.open('internet_flower.jpg')
    input_image=input_image.resize((224, 224)) 
    input_image=np.array(input_image)

    _,embedding=object_embedding(input_image)
    embedding =embedding.tolist()

    videos = {
    "vid1": "videos/How Green Roofs Can Help Cities  NPR.mp4",
    "vid2": "videos/What Does High-Quality Preschool Look Like  NPR Ed.mp4",
    "vid3": "videos/Why It’s Usually Hotter In A City  Lets Talk  NPR.mp4",
    }

    frame_infos = {}
    for vidId, video_path in videos.items():
        processed_frames_info = preprocess_video_frames(video_path)
        frame_infos[vidId] = processed_frames_info

    results = search_in_database(embedding)
    i=0
    for result in results:
        # print(result)
        i=i+1
        vidID, frameNum, bbox_info = result[0], result[1], result[-2]
        bbox_info =ast.literal_eval(bbox_info)
        video_info=frame_infos[vidID]
        frame_array = get_frame_by_num(frameNum,video_info)
        if frame_array is not None:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame_array = (frame_array * std) + mean
            frame_array*= 255.0
            drawn_frame=draw_bbox(frame_array, bbox_info)
            if np.all(drawn_frame == 0):
                print("Drawn image is all black.")
            cv2.imwrite(f'result_img/flower_internet/index_{i}.jpg', drawn_frame)
