#region initialization
import torch
import torch.nn as nn
from torch.nn import functional as F

eval_iters = 200
eval_interval = 500
block_size = 100
n_embd = 384
n_heads = 6
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.1
num_layer = 10
learning_rate = 3e-4
max_iters = 1


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
        wei = self.q(x) @ self.k(x).transpose(-2,-1) * self.k(x).shape[-1]**0.5
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
        self.position_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads) for _ in range(num_layer)] 
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx) + self.position_embedding(idx)
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
model.load_state_dict(torch.load("model.pth", map_location=device))
m = model.to(device)
print(sum((p.numel()) for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train_data']:.4f}, val loss {losses['test_data']:.4f}")

    logits, loss = model(*get_batch("train_data"))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
torch.save(model.state_dict(),"model.pth")

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
