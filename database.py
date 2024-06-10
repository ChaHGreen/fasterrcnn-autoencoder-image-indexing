import psycopg2
import csv
from embedding import object_embedding 
from preprocess import preprocess_video_frames
import numpy as np


db_config = {
    'dbname': 'assignment',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'db', 
    'port': '5432'
}

def insert_into_database(vid_id, frame_num, timestamp, detected_obj_id, label, score, bbox, embedding):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    insert_query = '''
    INSERT INTO detected_objects (vid_id, frame_num, timestamp, detected_obj_id, label, score, bbox_info, vector)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    '''

    try:
        cur.execute(insert_query, (vid_id, frame_num, timestamp, detected_obj_id, label, score, str(bbox), embedding))
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def process_video_and_csv(video_path, vid_id, csv_path):
    preprocessed_frames = preprocess_video_frames(video_path)
    frame_dict = {frame['frameNum']: frame for frame in preprocessed_frames}

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['vidId'] != vid_id:
                continue

            frame_num = int(row['frameNum'])
            frame_info = frame_dict.get(frame_num)
            if not frame_info:
                print(f"No matching frame found at frameNum {frame_num}")
                continue

            bbox = eval(row['bbox'])
            cropped_img = frame_info['frame'][int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

            _,embedding = object_embedding(cropped_img)
            embedding =embedding.tolist()
            insert_into_database(row['vidId'], frame_num, float(row['timestamp']), int(row['detectedObjId']), row['label'], float(row['score']), bbox, embedding)
            # print(f"Inserted: vidId={row['vidId']}, frameNum={frame_num}, timestamp={row['timestamp']}, detectedObjId={row['detectedObjId']}, label={row['label']}, score={row['score']}, bbox={bbox}, embedding={embedding[:5]}...")


if __name__ == "__main__":
    video_path = 'videos/Why It’s Usually Hotter In A City  Lets Talk  NPR.mp4'
    csv_path = 'detected_objects.csv'
    vid_id = 'vid3'
    process_video_and_csv(video_path, vid_id, csv_path)
    
