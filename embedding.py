import torch
from torchvision import transforms
from PIL import Image
from autoencoder import ConvAutoencoder
import cv2
from torchvision.transforms.functional import to_pil_image
import numpy as np

def object_embedding(cropped_img_np):
    # Initialize the auto encoder
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('checkpoint/autoencoder.pth', map_location=torch.device('cpu')))
    model.eval()

    # Make sure thw input is a numpy array
    if not isinstance(cropped_img_np, np.ndarray):
        raise ValueError("The cropped image must be a numpy array")

    # numpy --> PIL
    cropped_img = to_pil_image(cropped_img_np)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # reszie to match the model input size
        transforms.ToTensor(),          # PIL --> tensor
    ])

    # Apply Transform
    img_tensor = transform(cropped_img).unsqueeze(0)

    # Use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model.to(device)

    # Generate embedding
    with torch.no_grad():
        img_reconstruct,embedding = model(img_tensor) 

    embedding_np = embedding.cpu().numpy().flatten()

    return img_reconstruct,embedding_np