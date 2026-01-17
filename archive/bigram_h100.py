import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time
import urllib.request

# --- H100 Optimized Hyperparameters ---
batch_size = 128      # Massive throughput for A100/H100
block_size = 512      # Larger context window
max_iters = 5000      # 
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embed = 768         # GPT-2 Small scale width
num_heads = 12        # 768 / 12 = 64 head size
num_blocks = 12       # GPT-2 Small scale depth
dropout = 0.1
# --------------------------------------

# Set float32 matrix multiplication precision to 'high' for H100
torch.set_float32_matmul_precision('high')

print(f"Starting Bigram H100 training script...")
print(f"Device: {device}")

# --- Data Loading ---
dataset_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
input_file = 'input.txt'

if not os.path.exists(input_file):
    print(f"Downloading dataset from {dataset_url}...")
    urllib.request.urlretrieve(dataset_url, input_file)

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Use same mixed precision for evaluation
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Model Architecture ---

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(), # GELU is more common in modern Transformers
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads=num_heads) for _ in range(num_blocks)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Initialization ---
    model = BigramLanguageModel()
    model.to(device)

    # --- SPEED OPTIMIZATION: torch.compile ---
    if hasattr(torch, 'compile') and device == 'cuda':
        print("Compiling model for extra speedup...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    start_time = time.time()
    print(f"Training started at {time.ctime()}...")
    print(f"Hyperparameters: Batch={batch_size}, Block={block_size}, Embed={n_embed}, Blocks={num_blocks}")
    
    try:
        for iter in range(max_iters):

            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(model)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                # Intermediate Checkpoint Saving
                checkpoint_path = f'checkpoint_iter_{iter}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  > Saved checkpoint to {checkpoint_path}")

            xb, yb = get_batch('train')

            # Mixed Precision Training (using BFloat16 for H100)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    except Exception as e:
        print(f"\n[!] Training interrupted or crashed: {e}")
        print("Attempting to save emergency checkpoint...")
        torch.save(model.state_dict(), 'emergency_model_save.pth')
        print("Emergency save successful.")

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # --- Generation and Final Saving ---
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- Final Generation Sample ---")
    with torch.no_grad():
        generated_idx = model.generate(context, max_new_tokens=500)
        print(decode(generated_idx[0].tolist()))
    print("-------------------------------\n")

    save_path = 'shakespeare_h100_final.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    print("\n[Final Step] To download this file from Lambda Labs to your local machine, run:")
    print(f"scp ubuntu@<YOUR_VM_IP>:/home/ubuntu/{save_path} ./")
