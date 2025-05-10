import torch
import torch.nn as nn
from util import enwik8_string, dist, compute_compression
from util import enwik8
import matplotlib.pyplot as plt
import torch.nn.functional as F
import fire
import random
torch.manual_seed(42)

def batchify(data, length=100, batch_size=32):

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs = [data[start:start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target

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




class SelfAttention(nn.Module):
    def __init__(self, k=256, heads=4):

        super().__init__()
        self.k, self.heads = k, heads
        self.key = nn.Linear(k, k, bias=False)
        self.query = nn.Linear(k, k, bias=False)
        self.value = nn.Linear(k, k, bias=False)
        self.unify_heads = nn.Linear(k, k)

    def forward(self, x):

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

        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        dot.masked_fill(mask, float('-inf'))

        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, v).view(b, h, t, s)

        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        out = self.unify_heads(out)

        return out

class TransformerBlock(nn.Module):
  def __init__(self, k, heads):
    super().__init__()

    self.attention = SelfAttention(k, heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
      nn.Linear(k, 4 * k),
      nn.ReLU(),
      nn.Linear(4 * k, k))

  def forward(self, x):
    attended = self.attention(x)
    x = self.norm1(attended + x)

    fedforward = self.ff(x)
    return self.norm2(fedforward + x)

class Tranformer(nn.Module):
    def __init__(self, vocab_size, seq_length, k=256, heads=4, num_layers=4):

        super().__init__()
        assert k % heads == 0
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, k)
        self.pos_emb = nn.Embedding(seq_length, k)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(k, heads) for _ in range(num_layers)])
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(k, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        b, t, k = x.size()

        positions = self.pos_emb(torch.arange(t))[None, :, :].expand(b, t, k)

        x = x + positions

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.view(b*t, k)
        x = self.fc(x).view(b, t, self.vocab_size)
        return F.log_softmax(x, dim=2)
def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()
def sample_sequence(model, seed, max_context, length=600, temperature=0.5, verbose=False):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.

    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose: # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    print()
    return seed
def train_model(lr=0.001, k=256, heads=4, num_batches=32, context=100, batch_size=32, sample_length=600, test_subset=10000, test_batchsize=64):
    global train_loss
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    data_train, data_val, data_test = enwik8()
    model = Tranformer(vocab_size = 256, seq_length=500, k=k, heads=heads)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for i in range(num_batches):

        model.train()
        source, target = batchify(data_train, length=context, batch_size=batch_size)
        source, target = source.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(source)
        loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
        loss.backward()
        optimizer.step()

    if i != 0 and (i % 10 == 0):
        with torch.no_grad():

            model.eval()
            seedfr = random.randint(0, data_test.size(0) - context)
            seed = data_test[seedfr:seedfr + context].to(torch.long)

            if torch.cuda.is_available():
                seed = seed.cuda()

            sample_sequence(model, seed=seed, max_context=context, verbose=True, length=sample_length)

            upto = data_test.size(0) if i == num_batches - 1 else test_subset
            data_sub = data_test[:upto]

            bits_per_byte = compute_compression(model, data_sub, context=context,
                                                     batch_size=test_batchsize)

            print(f'epoch{i}: {bits_per_byte:.4} bits per byte')


if __name__ == '__main__':
    fire.Fire(train_model())