import csv
from collections import Counter
from preprocess import preprocess_video_frames
from detect_objects import detect_objects,load_pretrained_model

def main():
    videos = {
        "vid1": "videos/How Green Roofs Can Help Cities  NPR.mp4",
        "vid2": "videos/What Does High-Quality Preschool Look Like  NPR Ed.mp4",
        "vid3": "videos/Why It’s Usually Hotter In A City  Lets Talk  NPR.mp4",
    }
    output_csv = "detected_objects.csv"
    headers = ["vidId", "frameNum", "timestamp", "detectedObjId", "label", "score", "bbox"]

    detected_classes = Counter()

    with open(output_csv, 'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(headers)

        model = load_pretrained_model()

        for vidId, video_path in videos.items():
            print(vidId)
            processed_frames_info = preprocess_video_frames(video_path)
            detections = detect_objects(processed_frames_info, model)

            for det in detections:
                writer.writerow([vidId, det['frameNum'], det['timestamp'], det['detectedObjId'], det['label'], det['score'], det['bbox']])
                detected_classes[det['label']] += 1

    print(f"Detected {len(detected_classes)} unique classes across all videos.")
    print("Classes detected:", list(detected_classes.keys()))

if __name__ == "__main__":
    main()