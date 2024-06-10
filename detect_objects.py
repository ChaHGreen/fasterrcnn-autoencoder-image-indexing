import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
from torchvision.transforms import functional as F
from preprocess import preprocess_video_frames


# Load the pre-trained model and set it to evaluation mode
def load_pretrained_model():
    # model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = fasterrcnn_resnet50_fpn(pretrained=True).float()
    model.eval()
    return model

# Function to perform detection in one frame

def detect_frame(frame):
    model = load_pretrained_model()
    with torch.no_grad():
        detection = model(frame)
    prediction = detection[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    return boxes,labels,scores


# Function to perform object detections in videos
def detect_objects(processed_frames_info, model):
    detections = []
    for frame_info in processed_frames_info:
        # input format float32
        frame = frame_info['frame']
        frame_tensor = torch.tensor(frame, dtype=torch.float32)

        ## asjust the input channel [C, H, W]
        frame_tensor = frame_tensor.permute(2, 0, 1)

        # [C, H, W] --> [N, C, H, W]
        frame_tensor = frame_tensor.unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            prediction = model(frame_tensor)

        # Process prediction results
        frame_detections = process_predictions(prediction,frame_info['frameNum'], frame_info['timestamp'])
        detections.extend(frame_detections)

    return detections

# Utility function to process model predictions
def process_predictions(predictions, frameNum, timestamp):

    processed_results = []
    prediction = predictions[0]  # Assuming batch size of 1

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    detectedObjId = 0
    for box, label, score in zip(boxes, labels, scores):
        class_name = label_to_class_name(label)
        result = {
            'frameNum': frameNum,
            'timestamp': timestamp,
            'detectedObjId': detectedObjId,
            'label': class_name,
            'score': score.item(),
            'bbox': box.tolist(),
        }
        processed_results.append(result)
        detectedObjId += 1 

    return processed_results

def label_to_class_name(label):

    COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

    cls=COCO_INSTANCE_CATEGORY_NAMES[label]

    return cls

if __name__ == "__main__":
    # Example usage
    model = load_pretrained_model()
    video_path = "videos/How Green Roofs Can Help Cities  NPR.mp4"
    processed_frames_info = preprocess_video_frames(video_path, skip_frames=10, frame_size=(224, 224), scale=True, normalize=True)
    detections = detect_objects(processed_frames_info, model)
    print(detections)

