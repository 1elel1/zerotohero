#region initialization
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

eval_iters = 50
eval_interval = 250
block_size = 100
n_embd = 384
n_heads = 6
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True  # 用 TF32 做矩阵乘，RTX 5060 上每步从约 640ms 降到约 380ms
torch.backends.cudnn.allow_tf32 = True
dropout = 0.1
num_layer = 10
learning_rate = 3e-4
max_iters = 1  # 重新训练时用的是 6000（见 train_log_v2.txt）；平时聊天保持 1


with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text));                    vocab_size = len(chars)
stoi = { s:i for i, s in enumerate(chars)}
itos = { i:s for i, s in enumerate(chars)}
encoder = lambda s: [stoi[c] for c in s]
decoder = lambda arr: "".join(itos[n] for n in arr)

data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]
#endregion initialization

def get_batch(split):
    data = train_data if split == "train_data" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train_data','test_data']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        wei = self.q(x) @ self.k(x).transpose(-2,-1) * self.k(x).shape[-1]**-0.5  # 除以 sqrt(head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        out = wei @ self.v(x)
        return out 

class MultiHeads(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
        
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.multi_heads = MultiHeads(n_heads, head_size)
        self.ffn = FFN(n_embd)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_heads(self.ln(x))
        x = x + self.ffn(self.ln(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)  # 每个位置一个向量
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads) for _ in range(num_layer)] 
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)  # 位置 0..T-1，而不是 token id
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.ln(self.blocks(x))
        logits = self.lm(x)

        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1,logits.shape[-1]), targets.view(-1))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPT()
if os.path.exists("model.pth"):  # 没有存档就从头训练
    model.load_state_dict(torch.load("model.pth", map_location=device))
m = model.to(device)
print(sum((p.numel()) for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
best_val = float('inf')
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train_data']:.4f}, val loss {losses['test_data']:.4f}", flush=True)
        if losses['test_data'] < best_val:  # 只保存验证集最好的那次，防止过拟合后把好权重覆盖掉
            best_val = losses['test_data']
            torch.save(model.state_dict(), "model.pth")

    logits, loss = model(*get_batch("train_data"))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# context = torch.zeros((2,3), dtype = torch.long, device=device)
# while True:
#     idx = m.generate(context, 100)[0]
#     print(decoder(idx.tolist()[-100:]), end="")
#     # context = idx[-1]
#     context = idx.unsqueeze(0)   







context = torch.zeros((1,1), dtype=torch.long, device=device)  # 初始 BOS 或随便一个 token

while True:
    user_input = input("\n你：")
    if user_input.strip() == "":
        continue

    # 用户输入 → tokens
    user_tokens = encoder(user_input)
    user_tokens = torch.tensor(user_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # 拼接到上下文
    context = torch.cat([context, user_tokens], dim=1)

    # 模型生成
    idx = m.generate(context, max_new_tokens=100)

    # 取新增的部分
    new_tokens = idx[0].tolist()[len(context[0]):]

    # 输出模型回复
    print("模型：", decoder(new_tokens), end="")

    # 更新 context（继续累积）
    context = idx
