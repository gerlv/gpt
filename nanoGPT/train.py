from pathlib import Path

import numpy as np

dataset = Path('../data/tinyshakespear.txt')
text = dataset.read_text()

print(f"Dataset length: {len(text)}")
print(text[:100])

# tokenizer

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(f'Vocab size: {vocab_size}')

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]  # string to list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # list of integers to string

print(encode('hi there'))
print(decode(encode('hi there')))

# convert to pytorch tensor

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# split data into train and validation sets

n = int(0.9 * len(data))
train_data = data[:n]  # 90%
val_data = data[n:]  # 10%

# train on chunks of data
block_size = 8
print("block size", block_size, train_data[:block_size + 1])

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"When input is {context} the target: {target}")

# batching
torch.manual_seed(1337)
batch_size = 4  # how many independent sequences to process in parallel
block_size = 8  # maximum context window

# Q: what is torch.stack?
# A: torch.stack is a stack of multiple individual rows, so randint returns a stack of
#    multiple random values. And you can iterate on multiple values straight away

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print("inputs: ", xb.shape)
print(xb)
print("targets", yb.shape)
print(yb)

print('------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"When input is {context.tolist()} the target: {target}")

# Bigram model
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx)  # (B,T,C) - batch, time, vocab size
        # logits are scores for the next token, we are predicting what comes next
        # based on single token

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)  # or -1

            # wants B, C, T
            loss = F.cross_entropy(logits, targets)  # measures the quality of logits relative to targets
            # how well we are predicting

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # this is overkill in this case as bigram is only using the previous character

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # focus only on last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # B, T+1
        return idx

m = BigramLanguageModel(vocab_size=vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx=idx, max_new_tokens=100)[0].tolist()))

# train the model
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(100):
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print("Bigram Loss: ", loss.item())
print(decode(m.generate(idx=idx, max_new_tokens=100)[0].tolist()))

# math trick in self attention
# only focus on the previous tokens and take average of those tokens

import torch
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)

# We want x[b, t] = mean_{i<=t} x[b, i]
xbow = torch.zeros((B, T, C))  # bow = bag of words
for b in range(B):
    for t in range(T):
        xprev = x[b, : t+1]  # everything up to and including t'th token; shape - (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

print(x[0])
print(xbow[0])

# matrix multiplication
torch.manual_seed(42)

print(torch.tril(torch.ones(3, 3)))  # return a triangular of ones (on the left side)

a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print('a=', a)
print('b=', b)
print('c=', c)

# above self attention code can be replaced with, version 2
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
print(wei)
xbow2 = wei @ x  # (B, T, T) @ (B, T, C) --> (B, T, C)
print("allclose xbow2", torch.allclose(xbow, xbow2))

# version 3, softmax
tril = torch.tril(torch.ones(T, T))
print(f"tril", tril)
wei = torch.zeros((T, T))
print("wei", wei)
wei = wei.masked_fill(tril == 0, float('-inf'))
print("wei masked", wei)
wei = F.softmax(wei, dim=-1)
print("wei softmax", wei)
xbow3 = wei @ x
print("allclose xbow3", torch.allclose(xbow, xbow3))
