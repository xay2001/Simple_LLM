import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 超参数配置
class Config:
    def __init__(self):
        # 数据和模型配置
        self.batch_size = 4
        self.context_length = 16
        self.d_model = 64
        self.num_heads = 4
        self.num_blocks = 8
        self.dropout = 0.1
        
        # 训练配置
        self.max_iters = 5000
        self.eval_interval = 500
        self.learning_rate = 1e-3
        self.eval_iters = 200
        
        # 设备配置 - 支持Apple Silicon MPS
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

config = Config()

class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data) - self.max_length
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.max_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def load_data():
    """加载和预处理销售教材数据"""
    print("正在加载销售教材数据...")
    
    # 检查数据文件是否存在
    if not os.path.exists('sales_textbook.txt'):
        print("错误：找不到 sales_textbook.txt 文件")
        return None, None, None
    
    # 读取文本数据
    with open('sales_textbook.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"数据加载完成，总字符数: {len(text)}")
    
    # 使用GPT-3兼容的tokenizer
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    # 对文本进行tokenization
    tokens = enc.encode(text)
    print(f"Token化完成，总token数: {len(tokens)}")
    
    # 数据分割：90%训练，10%验证
    split_idx = int(0.9 * len(tokens))
    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]
    
    print(f"训练数据: {len(train_data)} tokens")
    print(f"验证数据: {len(val_data)} tokens")
    
    return train_data, val_data, enc

class FeedForward(nn.Module):
    """前馈神经网络模块"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.ffn(x)

class Attention(nn.Module):
    """单头注意力机制"""
    def __init__(self, d_model, head_size, context_length, dropout=0.1):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 注册因果掩码（下三角矩阵）
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x):
        B, T, C = x.shape
        
        # 计算查询、键、值
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        # 计算注意力分数
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)
        
        # 应用因果掩码
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax归一化
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # 应用注意力权重
        out = wei @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, context_length, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.head_size = d_model // num_heads
        self.heads = nn.ModuleList([
            Attention(d_model, self.head_size, context_length, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 并行计算所有注意力头
        head_outputs = [head(x) for head in self.heads]
        
        # 拼接所有头的输出
        out = torch.cat(head_outputs, dim=-1)
        
        # 投影和dropout
        out = self.proj(out)
        out = self.dropout(out)
        return out

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, d_model, num_heads, context_length, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, num_heads, context_length, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # 自注意力 + 残差连接
        x = x + self.sa(self.ln1(x))
        # 前馈网络 + 残差连接
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    """完整的Transformer语言模型"""
    def __init__(self, vocab_size, d_model, num_heads, num_blocks, context_length, dropout=0.1):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        
        # 嵌入层
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(context_length, d_model)
        
        # Transformer块堆叠
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, context_length, dropout)
            for _ in range(num_blocks)
        ])
        
        # 最终层归一化和线性输出层
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token嵌入和位置嵌入
        tok_emb = self.token_embedding_table(idx)  # (B, T, d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)
        
        # 通过Transformer块
        x = self.blocks(x)  # (B, T, d_model)
        x = self.ln_f(x)    # (B, T, d_model)
        
        # 生成logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """生成新的token序列"""
        for _ in range(max_new_tokens):
            # 限制序列长度到context_length
            idx_cond = idx[:, -self.context_length:]
            
            # 获取预测
            logits, loss = self(idx_cond)
            
            # 只关注最后一个时间步的logits
            logits = logits[:, -1, :]  # (B, C)
            
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)  # (B, C)
            
            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # 拼接到序列后面
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters):
    """估算训练和验证损失"""
    out = {}
    model.eval()
    
    for split, dataloader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k, (xb, yb) in enumerate(dataloader):
            if k >= eval_iters:
                break
            xb, yb = xb.to(config.device), yb.to(config.device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

def train_model():
    """训练模型主函数"""
    print("开始训练模型...")
    
    # 加载数据
    train_data, val_data, tokenizer = load_data()
    if train_data is None:
        return None, None
    
    vocab_size = tokenizer.n_vocab
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_data, tokenizer, config.context_length)
    val_dataset = TextDataset(val_data, tokenizer, config.context_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 创建模型
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        context_length=config.context_length,
        dropout=config.dropout
    ).to(config.device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    print(f"在 {config.device} 上开始训练...")
    
    for step in range(config.max_iters):
        # 每隔一定步数评估模型
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, config.eval_iters)
            print(f"步数 {step}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())
        
        # 获取一个批次的数据
        try:
            if 'train_iter' not in locals():
                train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        
        xb, yb = xb.to(config.device), yb.to(config.device)
        
        # 前向传播
        logits, loss = model(xb, yb)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print("训练完成！")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    steps = [i * config.eval_interval for i in range(len(train_losses))]
    plt.plot(steps, train_losses, label='training loss', marker='o')
    plt.plot(steps, val_losses, label='validation loss', marker='s')
    plt.xlabel('Training steps')
    plt.ylabel('loss')
    plt.title('The change of loss during the training process')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt="", max_new_tokens=100):
    """生成文本"""
    model.eval()
    
    # 如果没有提供prompt，使用随机起始
    if not prompt:
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    else:
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.device).unsqueeze(0)
    
    # 生成文本
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens)
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text

def main():
    """主函数"""
    print("=" * 60)
    print("模块化 Transformer 语言模型")
    print("=" * 60)
    print(f"配置信息:")
    print(f"  - 批次大小: {config.batch_size}")
    print(f"  - 上下文长度: {config.context_length}")
    print(f"  - 模型维度: {config.d_model}")
    print(f"  - 注意力头数: {config.num_heads}")
    print(f"  - Transformer块数: {config.num_blocks}")
    print(f"  - 最大训练步数: {config.max_iters}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 设备: {config.device}")
    print("=" * 60)
    
    # 训练模型
    model, tokenizer = train_model()
    
    if model is not None:
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__
        }, 'transformer_model.pth')
        print("模型已保存为 transformer_model.pth")
        
        # 生成示例文本
        print("\n" + "=" * 60)
        print("生成文本示例:")
        print("=" * 60)
        
        # 示例1：无提示生成
        print("1. 无提示生成:")
        generated_text = generate_text(model, tokenizer, "", 50)
        print(generated_text)
        print()
        
        # 示例2：带提示生成
        print("2. 带提示生成 (提示: 'Sales'):")
        generated_text = generate_text(model, tokenizer, "Sales", 50)
        print(generated_text)
        print()
        
        # 示例3：带提示生成
        print("3. 带提示生成 (提示: 'Building rapport'):")
        generated_text = generate_text(model, tokenizer, "Building rapport", 50)
        print(generated_text)
        
        print("=" * 60)
        print("训练和推理完成！")

if __name__ == "__main__":
    main() 