#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化Transformer语言模型 - 模型定义

包含所有模型组件和配置类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from typing import Optional

class Config:
    """模型配置类"""
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
    
    def __repr__(self):
        return f"Config(d_model={self.d_model}, num_heads={self.num_heads}, num_blocks={self.num_blocks}, device='{self.device}')"

class FeedForward(nn.Module):
    """前馈神经网络模块
    
    实现标准的Transformer前馈网络：
    Linear -> ReLU -> Linear -> Dropout
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        
        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        return self.ffn(x)

class Attention(nn.Module):
    """单头注意力机制
    
    实现缩放点积注意力（Scaled Dot-Product Attention）
    """
    def __init__(self, d_model: int, head_size: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 注册因果掩码（下三角矩阵）
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        
        Returns:
            注意力输出 (batch_size, seq_len, head_size)
        """
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
    """多头注意力机制
    
    并行运行多个注意力头，然后拼接结果
    """
    def __init__(self, d_model: int, num_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        
        # 创建多个注意力头
        self.heads = nn.ModuleList([
            Attention(d_model, self.head_size, context_length, dropout)
            for _ in range(num_heads)
        ])
        
        # 输出投影层
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        
        Returns:
            多头注意力输出 (batch_size, seq_len, d_model)
        """
        # 并行计算所有注意力头
        head_outputs = [head(x) for head in self.heads]
        
        # 拼接所有头的输出
        out = torch.cat(head_outputs, dim=-1)  # (B, T, d_model)
        
        # 投影和dropout
        out = self.proj(out)
        out = self.dropout(out)
        return out

class TransformerBlock(nn.Module):
    """Transformer块
    
    包含多头注意力和前馈网络，以及残差连接和层归一化
    """
    def __init__(self, d_model: int, num_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, num_heads, context_length, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        
        Returns:
            Transformer块输出 (batch_size, seq_len, d_model)
        """
        # 自注意力 + 残差连接
        x = x + self.sa(self.ln1(x))
        # 前馈网络 + 残差连接
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    """完整的Transformer语言模型
    
    包含token嵌入、位置嵌入、多层Transformer块和语言建模头
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_blocks: int, 
                 context_length: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length
        
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
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"模型参数数量: {total_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Args:
            idx: 输入token索引 (batch_size, seq_len)
            targets: 目标token索引 (batch_size, seq_len), 可选
        
        Returns:
            logits: 输出logits (batch_size, seq_len, vocab_size)
            loss: 损失值（如果提供了targets）
        """
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
        
        # 计算损失
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """生成新的token序列
        
        Args:
            idx: 起始序列 (batch_size, seq_len)
            max_new_tokens: 要生成的新token数量
            temperature: 温度参数，控制随机性
        
        Returns:
            生成的完整序列 (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # 限制序列长度到context_length
            idx_cond = idx[:, -self.context_length:]
            
            # 获取预测
            with torch.no_grad():
                logits, _ = self(idx_cond)
            
            # 只关注最后一个时间步的logits
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # 拼接到序列后面
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def save_pretrained(self, save_path: str, config: Config):
        """保存模型和配置"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config.__dict__,
            'vocab_size': self.vocab_size,
            'model_args': {
                'd_model': self.d_model,
                'context_length': self.context_length
            }
        }, save_path)
        print(f"模型已保存到: {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str):
        """从保存的文件加载模型
        
        Args:
            load_path: 模型文件路径
        
        Returns:
            model: 加载的模型
            config: 模型配置
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # 重建配置
        config = Config()
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 创建模型
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_blocks=config.num_blocks,
            context_length=config.context_length,
            dropout=0.0  # 推理时不使用dropout
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"模型已从 {load_path} 加载完成")
        return model, config

def generate_text(model: TransformerLanguageModel, tokenizer, prompt: str = "", 
                 max_new_tokens: int = 100, temperature: float = 1.0, device: str = 'cpu') -> str:
    """生成文本的便捷函数
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 生成的最大token数
        temperature: 温度参数
        device: 设备
    
    Returns:
        生成的文本
    """
    model.eval()
    model.to(device)
    
    # 准备输入
    if not prompt:
        # 空提示，使用随机起始token
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        # 编码提示
        tokens = tokenizer.encode(prompt)
        context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # 生成文本
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text

# 创建默认配置实例
default_config = Config()

if __name__ == "__main__":
    # 测试模型创建
    print("测试模型组件...")
    
    config = Config()
    print(f"配置: {config}")
    
    # 获取tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    vocab_size = tokenizer.n_vocab
    
    # 创建模型
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        context_length=config.context_length,
        dropout=config.dropout
    )
    
    print(f"模型创建成功，参数量: {model.get_num_params()/1e6:.2f}M")
    
    # 测试前向传播
    test_input = torch.randint(0, vocab_size, (2, 8))  # batch_size=2, seq_len=8
    logits, loss = model(test_input, test_input)
    print(f"前向传播测试: 输入形状{test_input.shape} -> 输出形状{logits.shape}")
    
    # 测试生成
    generated = model.generate(test_input[:1], max_new_tokens=5)
    print(f"生成测试: 输入长度{test_input.shape[1]} -> 生成长度{generated.shape[1]}")
    
    print("所有测试通过！") 