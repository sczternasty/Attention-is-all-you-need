"""
IMDB Sentiment Analysis using Custom Self-Attention Mechanisms

CV-Ready Implementation: Advanced NLP with Custom Attention Architectures
=======================================================================

This module demonstrates advanced machine learning skills by implementing:
- Custom self-attention mechanisms from scratch for sentiment analysis
- Multi-head attention with configurable dimensions and heads
- Efficient batching strategies for variable-length sequences
- Custom training loops with comprehensive evaluation metrics

Key Technical Skills Demonstrated:
- Deep understanding of transformer architectures and attention mechanisms
- Advanced PyTorch implementation without relying on pre-built libraries
- Custom neural network design and optimization
- Research-level implementation of state-of-the-art NLP techniques

Author: [Your Name]
Date: [Current Date]
CV Category: Natural Language Processing, Sentiment Analysis, Custom Attention Implementation
"""

import torch
import torch.nn as nn
from data_download import load_imdb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import fire

# Set random seed for reproducibility
torch.manual_seed(42)

def load_data(max_token=None):
    """
    Load and preprocess IMDB dataset for sentiment analysis.
    
    CV Skill: Data Pipeline Development & Preprocessing
    - Efficient data loading and batching strategies
    - Dynamic padding for variable-length sequences
    - Vocabulary management and tokenization
    
    Args:
        max_token (int, optional): Maximum number of tokens per batch. 
                                 If None, uses the maximum sequence length.
    
    Returns:
        tuple: ((train_X, train_y), (val_X, val_y), vocab_size)
               - train_X: Training data batches
               - train_y: Training labels
               - val_X: Validation data batches  
               - val_y: Validation labels
               - vocab_size: Size of vocabulary
    """
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
    train_X, train_y = batchify(x_train, y_train, w2i, max_token)
    val_X, val_y = batchify(x_val, y_val, w2i, max_token)
    return (train_X, train_y), (val_X, val_y), len(w2i)

def batchify(X, y, w2i, max_token=None):
    """
    CV Skill: Advanced Batching & Memory Optimization
    - Custom batching strategy for variable-length sequences
    - Memory-efficient token management
    - Dynamic padding implementation
    
    Creates efficient batches while respecting token limits and memory constraints.
    """

    if not max_token:
        max_token = max([len(x) for x in X])
    batched_x = []
    batched_y = []
    curr_batch_x = []
    curr_batch_y = []
    curr_batch_size = 0
    for i in reversed(range(len(X))):
        curr = X[i]
        curr_y = y[i]
        if curr_batch_size + len(
                curr) > max_token:  # Adding the current instance to the batch would exceed the max token limit
            batch_max = max([len(x) for x in curr_batch_x])
            curr_batch_x = [instance + [w2i[".pad"]] * (batch_max - len(instance)) for instance in curr_batch_x]
            batched_x.append(curr_batch_x)
            batched_y.append(curr_batch_y)
            curr_batch_x = [curr]
            curr_batch_y = [curr_y]
            curr_batch_size = len(curr)
        elif curr_batch_size + len(
                curr) == max_token:  # Adding the current instance to the batch would exactly match the max token limit
            curr_batch_x.append(curr)
            curr_batch_y.append(curr_y)
            batch_max = max([len(x) for x in curr_batch_x])
            curr_batch_x = [instance + [w2i[".pad"]] * (batch_max - len(instance)) for instance in curr_batch_x]
            batched_x.append(curr_batch_x)
            batched_y.append(curr_batch_y)
            curr_batch_x = []
            curr_batch_y = []
            curr_batch_size = 0
        else:  # Adding the current instance to the batch would not exceed the max token limit
            curr_batch_x.append(curr)
            curr_batch_y.append(curr_y)
            curr_batch_size += len(curr)

    if curr_batch_x:  # If there are any remaining instances in the current batch
        batch_max = max(len(x) for x in curr_batch_x)
        padded_x = [x + [w2i[".pad"]] * (batch_max - len(x)) for x in curr_batch_x]
        batched_x.append(padded_x)
        batched_y.append(curr_batch_y)

    train_X = [torch.tensor(batch, dtype=torch.long) for batch in batched_x]
    train_y = [torch.tensor(batch, dtype=torch.long) for batch in batched_y]

    return train_X, train_y

def accuracy(pred, y):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == y).float()
    acc = correct.sum() / len(correct)
    return acc
def plot_loss(trainlosses, testlosses=None, labels = None):
    plt.plot(trainlosses, label=labels[0], color='blue')
    if testlosses:
        plt.plot(testlosses, label=labels[1], color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.show()

class SimpleSelfAttention(nn.Module):
    """
    Simple Self-Attention model for text classification.
    
    This model implements a basic self-attention mechanism where:
    1. Input tokens are embedded into a k-dimensional space
    2. Self-attention weights are computed using dot-product attention
    3. Global pooling is applied to aggregate sequence information
    4. Final classification is performed using a linear layer
    
    Args:
        vocab_size (int): Size of the vocabulary
        k (int): Embedding dimension and hidden size
    """
    
    def __init__(self, vocab_size, k):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, k)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(k, 2)
    
    def forward(self, x):
        """
        Forward pass through the self-attention model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        # x size = (batch_size, seq_len)
        x = self.emb(x)  # x size = (batch_size, seq_len, k)
        
        # Compute self-attention weights
        raw_weights = torch.bmm(x, x.transpose(1, 2))
        weights = F.softmax(raw_weights, dim=2)
        
        # Apply attention weights
        x = torch.bmm(weights, x)
        x = x.transpose(1, 2)  # x size = (batch_size, k, seq_len)
        
        # Global pooling and classification
        x = self.global_pool(x)  # x size = (batch_size, k, 1)
        x = x.squeeze(2)  # x size = (batch_size, k)
        x = self.fc(x)  # x size = (batch_size, 2)
        
        return x


class SelfAttention(nn.Module):
    def __init__(self, vocab_size, k=256, heads=4):

        super().__init__()
        self.k, self.heads = k, heads
        assert k % heads == 0
        self.key = nn.Linear(k, k, bias=False)
        self.query = nn.Linear(k, k, bias=False)
        self.value = nn.Linear(k, k, bias=False)
        self.unify_heads = nn.Linear(k, k)

        self.emb = nn.Embedding(vocab_size, k)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(k, 2)

    def forward(self, x):

        x = self.emb(x)

        b, t, e = x.size()
        h = self.heads

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        s = e // h

        k = k.view(b, t, h, s)
        q = q.view(b, t, h, s)
        v = v.view(b, t, h, s)

        k = k.transpose(1, 2).contiguous().view(b * h, t, s)
        q = q.transpose(1, 2).contiguous().view(b * h, t, s)
        v = v.transpose(1, 2).contiguous().view(b * h, t, s)

        dot = torch.bmm(q, k.transpose(1, 2))

        dot = dot / (s ** (1 / 2))

        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, v).view(b, h, t, s)

        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        out = self.unify_heads(out)

        out = out.transpose(1, 2)
        out = self.global_pool(out)
        out = out.squeeze(2)
        out = self.fc(out)
        return out




def train_model(epochs=20, lr=0.001, k=256, max_token=None, heads=4):
    global train_loss
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    (train_X, train_y), (val_X, val_y), vocab_size = load_data(max_token=max_token)
    model = SelfAttention(vocab_size = vocab_size, k=k, heads=heads)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        train_total_loss = 0
        val_total_loss = 0
        train_acc = 0
        val_acc = 0

        model.train()
        for i, (x, y) in enumerate(zip(train_X, train_y)):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += accuracy(output, y).item()
        train_loss = train_total_loss / len(train_X)
        train_losses.append(train_loss)
        train_acc /= len(train_X)

        model.eval()
        with torch.no_grad():
            for x, y in zip(val_X, val_y):
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_total_loss += loss.item()
                val_acc += accuracy(output, y).item()
        val_loss = val_total_loss / len(val_X)
        val_losses.append(val_loss)
        val_acc /= len(val_X)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_acc * 100:.2f}%, Val Loss: {val_loss}, Val Accuracy: {val_acc * 100:.2f}%")
    plot_loss(train_losses, val_losses, labels=['Train Loss', 'Validation Loss'])

if __name__ == '__main__':
    fire.Fire(train_model)