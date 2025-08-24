import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

def remove_random_patches(images, patch_size=8):

    B, C, H, W = images.size()
    images = images.clone()
    mask = torch.ones_like(images)
    targets = torch.zeros(B, C, patch_size, patch_size)
    patch_positions = []

    for i in range(B):
        x = random.randint(0, H - patch_size)
        y = random.randint(0, W - patch_size)
        targets[i] = images[i, :, x:x+patch_size, y:y+patch_size].clone()
        images[i, :, x:x+patch_size, y:y+patch_size] = 0.0
        mask[i, :, x:x+patch_size, y:y+patch_size] = 0
        patch_positions.append((x, y))

    return images, mask, targets, patch_positions

def show_image(image):

    image = image * 0.5 + 0.5
    image = image.permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.show()

class SelfAttention(nn.Module):
    def __init__(self, emb_size=256, heads=4, dropout=0.1):
        super().__init__()
        self.emb_size, self.heads = emb_size, heads
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.unify_heads = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        H = self.heads

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        S = C // H

        k = k.view(B, T, H, S)
        q = q.view(B, T, H, S)
        v = v.view(B, T, H, S)

        k = k.transpose(1, 2).contiguous().view(B * H, T, S)
        q = q.transpose(1, 2).contiguous().view(B * H, T, S)
        v = v.transpose(1, 2).contiguous().view(B * H, T, S)

        dot = torch.bmm(q, k.transpose(1, 2))
        dot = dot / math.sqrt(S)
        dot = F.softmax(dot, dim=2)
        dot = self.dropout(dot)
        out = torch.bmm(dot, v).view(B, H, T, S)
        out = out.transpose(1, 2).contiguous().view(B, T, S * H)
        out = self.unify_heads(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3, emb_size=256, heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        assert emb_size % heads == 0
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.heads = heads
        grid_size = img_size // patch_size

        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_size))

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(emb_size, heads, dropout) for _ in range(num_layers)]
        )

        self.output_proj = nn.ConvTranspose2d(emb_size, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.size()
        

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        

        x = x + self.pos_embedding

        x = self.transformer_blocks(x)

        x = x.transpose(1, 2).view(B, self.emb_size, H//self.patch_size, W//self.patch_size)
        x = self.output_proj(x)
        
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
            masked_images, _, targets, patch_positions = remove_random_patches(images)
            masked_images = masked_images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(masked_images)

            predicted_patches = torch.zeros_like(targets)
            for i, (x, y) in enumerate(patch_positions):
                predicted_patches[i] = outputs[i, :, x:x+8, y:y+8]
            
            loss = criterion(predicted_patches, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

def evaluate_model(model, test_loader, num_examples=5, device='cuda'):
    model.eval()
    with torch.no_grad():

        images, _ = next(iter(test_loader))
        images = images[:num_examples].to(device)

        masked_images, _, targets, patch_positions = remove_random_patches(images)

        outputs = model(masked_images)

        plt.figure(figsize=(15, 3*num_examples))
        
        for i in range(num_examples):

            plt.subplot(num_examples, 3, i*3 + 1)
            show_image(images[i].cpu())
            plt.title('Original')

            plt.subplot(num_examples, 3, i*3 + 2)
            show_image(masked_images[i].cpu())
            plt.title('Masked')

            plt.subplot(num_examples, 3, i*3 + 3)
            show_image(outputs[i].cpu())
            plt.title('Reconstructed')
        
        plt.tight_layout()
        plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(
        img_size=32,
        patch_size=8,
        in_channels=3,
        emb_size=256,
        heads=8,
        num_layers=6
    )
    train_model(model, train_loader, num_epochs=10, device=device)
    

    torch.save(model.state_dict(), 'cifar_inpainting_model.pth')
    

    print("\nEvaluating model on test set...")
    evaluate_model(model, test_loader, num_examples=5, device=device)

if __name__ == '__main__':
    main() 