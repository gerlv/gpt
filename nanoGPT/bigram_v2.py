from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

batch_size = 32
block_size = 8
training_split = 0.9
steps = 1000
max_new_tokens = 7
eval_iters = 200
device = 'cpu'
n_embed = 32

dataset = Path('../data/tinyshakespear.txt')
text = dataset.read_text()

print(f"Dataset length: {len(text)}")

# tokenizer

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Vocab', ''.join(chars))
print(f'Vocab size: {vocab_size}')

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]  # string to list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # list of integers to string

# convert to pytorch tensor

data = torch.tensor(encode(text), dtype=torch.long)

# split data into train and validation sets

n = int(training_split * len(data))
train_data = data[:n]  # 90%
val_data = data[n:]  # 10%

# train on chunks of data

# Q: what is torch.stack?
# A: torch.stack is a stack of multiple individual rows, so randint returns a stack of
#    multiple random values. And you can iterate on multiple values straight away


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Bigram model


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        print(B, T)

        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        # x holds token identities and positions where these tokens occur
        logits = self.lm_head(x)  # (B, T, vocab_size) - batch, time, vocab size

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


model = BigramLanguageModel()
m = model.to(device=device)

loss = None

# train the model
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for step in range(steps):
    if step % eval_iters == 0:
        losses = estimate_loss()
        print(f"step ({step}): train loss {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Bigram Loss: ", loss.item())
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx=idx, max_new_tokens=max_new_tokens)[0].tolist()))
