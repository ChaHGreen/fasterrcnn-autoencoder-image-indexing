import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from pycocotools.coco import COCO
from dataset import CocoObjectsCropDataset

class ConvAutoencoder(nn.Module):
    def __init__(self, embedding_size=32):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [batch, 32, 112, 112]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [batch, 64, 56, 56]
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [batch, 128, 28, 28]
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # [batch, 256, 14, 14]
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # [batch, 512, 7, 7]
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_size)  # Fully connected layer to generate embeding
        )

        self.decoder_input = nn.Linear(embedding_size, 512 * 7 * 7)  # Input to decoder

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # [batch, 256, 14, 14]
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [batch, 128, 28, 28]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 64, 56, 56]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 32, 112, 112]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # [batch, 3, 224, 224]
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        x = self.decoder_input(embedding)
        x = x.view(-1, 512, 7, 7)  # reshape the embedding to match the input of conv layer
        x = self.decoder(x)
        return x, embedding  # Reconstructed image and embedding


def train_autoencoder(dataset, epochs=10, batch_size=32, learning_rate=1e-3):
    # gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ConvAutoencoder().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("epoch: ",epoch)
        model.train()
        train_loss = 0.0
        for data in dataloader:
            inputs = data.to(device)
            optimizer.zero_grad()
            
            # forward
            outputs,_ = model(inputs)   ## embedding is not needed right now
            
            # loss
            loss = criterion(outputs, inputs)
            
            # back
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
        with open("training_log.txt", "a") as f:
            f.write(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}\n')
        if epoch % 10==0:
            torch.save(model.state_dict(), '/workspaces/artificial_intelligence/checkpoint/autoencoder_{}.pth'.format(epoch))
    
    # Save the model
    torch.save(model.state_dict(), '/workspaces/artificial_intelligence/checkpoint/autoencoder.pth')

    print('Training complete')
    


if __name__ == "__main__":
    classes_of_interest=['couch', 'bowl', 'sink', 'dining table', 'car', 'cat', 'traffic light', 'person', 'sheep', 'bottle', 'parking meter', 'clock', 'microwave', 'cup', 'sports ball', 'bird', 'tv', 'oven', 'chair', 'toilet', 'truck', 'wine glass', 'bench', 'teddy bear', 'train', 'bed', 'bicycle', 'cell phone', 'hair drier', 'vase', 'apple', 'umbrella', 'tennis racket', 'motorcycle', 'scissors', 'refrigerator', 'fire hydrant', 'remote', 'tie']
    coco_root = '/workspaces/coco/train2017'
    annFile = '/workspaces/coco/annotations/instances_train2017.json'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print("Start loading COCO images")
    dataset = CocoObjectsCropDataset(coco_root, annFile, classes_of_interest,  subset_size=5000, transform=transform)
    
    print("Start Traning")
    train_autoencoder(dataset,epochs=30, batch_size=32, learning_rate=1e-3)
