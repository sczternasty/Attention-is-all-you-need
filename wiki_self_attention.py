import torch
import torch.nn as nn
import tqdm
import math
import torch.nn.functional as F
import fire
import random
import os
import numpy as np
import gzip
from torch.optim.lr_scheduler import LambdaLR
torch.manual_seed(42)

def enwik8(path=None, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    cwd = os.getcwd()
    if path is None:
        path = cwd + '\\data\\enwik8.gz'

    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX.copy()), torch.from_numpy(vaX.copy()), torch.from_numpy(teX.copy())

def compute_compression(model, data, context, batch_size, verbose=False, tok=None, skip=0, device='cpu'):


    """
    Compute the _compression_ of a dataset under a model. That is, given a model, in how many bits could we represent
    the dataset. This requires us to turn a given probability distribution into a code for the outcomes.

    See [this video](https://youtu.be/mSneVjDvzNQ) for an explanation.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the  data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """
    LOG2E = math.log2(math.e)
    LOGE2 = math.log(2.0)
    bits, tot = 0.0, 0
    batch = []
    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.

    target_indices = []
    i, ic = 0, 0

    for current in tqdm.trange(skip, data.size(0)) if verbose else range(skip, data.size(0)):

        # `current` is the character which we will ultimately predict

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        # if tok is not None:
        #     print(instance[:-1], tok.decode(instance[:-1]))
        #     print(instance[-1:], tok.decode(instance[-1:]))

        target_indices.append(instance.size(0) - 2) # index of the last element of the input to the model

        if instance.size(0) < context + 1:
            assert skip < context # We shouldn't get here if we skip the first `context` characters

            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or current == data.size(0) - 1:
            # batch is full or we are at the last instance, run it through the model

            b = len(batch)

            ti = torch.tensor(target_indices) + 1
            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1] # input
            target = all[torch.arange(b), ti]  # target values

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(output.logits, dim=2) # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context, -1)}'

            lnprobs = output[torch.arange(b, device=device), target_indices, target]
            log2probs = lnprobs / LOGE2
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.



            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilities) to the running total
            batch, target_indices = [], []  # clear the buffer

    if isinstance(bits, torch.Tensor):
        bits = bits.item()

    return bits

def sample_batch(data, length=100, batch_size=32):
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    X = [data[start:start + length] for start in starts]
    y = [data[start + 1:start + length + 1] for start in starts]

    X = torch.cat([s.unsqueeze(dim=0) for s in X], dim=0).to(torch.long)
    y = torch.cat([s.unsqueeze(dim=0) for s in y], dim=0).to(torch.long)

    return X, y

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

        dot = torch.bmm(q, k.transpose(1, 2)) # B * H, T, S) @ (B * H, S, T) -> (B * H, T, T)

        dot = dot / math.sqrt(S)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        dot = dot.masked_fill(mask == 1, float('-inf'))

        dot = F.softmax(dot, dim=2)

        dot = self.dropout(dot)
        out = torch.bmm(dot, v).view(B, H, T, S)

        out = out.transpose(1, 2).contiguous().view(B, T, S * H) # (B, H, T, S) -> (B, T, S * H) -> (B, T, C)

        out = self.unify_heads(out)
        [[]]
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
      nn.Dropout(dropout))

  def forward(self, x):
    x = x + self.attention(self.norm1(x))
    x = x + self.ff(self.norm2(x))
    return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_length, emb_size=256, heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        assert emb_size % heads == 0
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.heads = heads

        # Embeddings
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(seq_length, emb_size)
        
        # Dropout layers
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_dropout = nn.Dropout(dropout)
        self.final_dropout = nn.Dropout(dropout)

        # Transformer blocks with dropout parameter
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_size, heads, dropout=dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.emb_dropout(x)
        
        B, T, C = x.size()
        positions = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(0).expand(B, T, C)
        positions = self.pos_dropout(positions)
        
        x = x + positions

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = self.final_dropout(x)
        x = x.view(B*T, C)
        x = self.fc(x).view(B, T, self.vocab_size)
        return F.log_softmax(x, dim=2)

def sample(probs, temp=1.0):

    if temp == 0.0:
        return probs.argmax()

    p = F.softmax(probs / temp, dim=0)
    out = torch.multinomial(p, 1)

    return out

def sample_sequence(model, seed, max_context, length=600, temp=0.5, verbose=False, device='cpu'):

    seq = seed.detach().clone()

    if verbose:
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    model = model.to(device)
    seq = seq.to(device)
    for _ in range(length):

        X = seq[-max_context:]
        output = model(X.unsqueeze(0))

        c = sample(output[0, -1, :], temp)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        seq = torch.cat([seq, c], dim=0)

    print()
    return seed





def get_lr_schedule(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0 / math.sqrt(step)

    return LambdaLR(optimizer, lr_lambda)

def train_model(lr=3e-4, k=384, heads=6, num_layers=6, num_batches=10000, context=256, batch_size=64, sample_length=600, test_subset=10000, test_batchsize=64, warmup_steps=4000):

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data_train, data_val, data_test = enwik8()

    model = Transformer(vocab_size = 256, seq_length=500, emb_size=k, heads=heads, num_layers=num_layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_lr_schedule(optimizer, warmup_steps)
    for i in range(num_batches):
        X, y = sample_batch(data_train, length=context, batch_size=batch_size)
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = F.nll_loss(output.transpose(2, 1), y, reduction='mean')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if i != 0 and (i % 1000 == 0):
            with torch.no_grad():

                model.eval()
                context_seed = random.randint(0, data_test.size(0) - context)
                chunk = data_test[context_seed:context_seed + context].to(torch.long)
                chunk.to(device)

                sample_sequence(model, seed=chunk, max_context=context, verbose=True, length=sample_length, device=device)

                limit = data_test.size(0) if i == num_batches - 1 else test_subset
                data_sub = data_test[:limit]

                bits_per_byte = compute_compression(model, data_sub, context=context, batch_size=test_batchsize, device=device)

                print(f'Batch{i+1}/{num_batches}: {bits_per_byte:.4} bits per byte')


if __name__ == '__main__':
    fire.Fire(train_model)