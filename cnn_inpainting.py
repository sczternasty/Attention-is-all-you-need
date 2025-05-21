import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

# Data loading and preprocessing
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

def remove_random_patches(images, patch_size=7):
    B, C, H, W = images.size()
    images = images.clone()
    mask = torch.ones_like(images)
    targets = torch.zeros(B, C, patch_size, patch_size)

    for i in range(B):
        x = random.randint(0, H - patch_size)
        y = random.randint(0, W - patch_size)
        targets[i] = images[i, :, x:x+patch_size, y:y+patch_size].clone()
        images[i, :, x:x+patch_size, y:y+patch_size] = 0.0
        mask[i, :, x:x+patch_size, y:y+patch_size] = 0

    return images, mask, targets

def show_image(image):
    image = image.squeeze().numpy()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

class CNNInpainting(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model(model, train_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            masked_images, _, targets = remove_random_patches(images)
            masked_images = masked_images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(masked_images)
            
            # Extract the predicted patch from the output
            B, C, H, W = outputs.size()
            predicted_patches = torch.zeros_like(targets)
            for i in range(B):
                x = random.randint(0, H - 7)
                y = random.randint(0, W - 7)
                predicted_patches[i] = outputs[i, :, x:x+7, y:y+7]
            
            loss = criterion(predicted_patches, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNInpainting()
    train_model(model, train_loader, num_epochs=10, device=device)
    
    # Save the model
    torch.save(model.state_dict(), 'cnn_inpainting_model.pth')

if __name__ == '__main__':
    main() 