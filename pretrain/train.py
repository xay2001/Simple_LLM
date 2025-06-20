#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化Transformer语言模型 - 训练脚本

包含数据加载、训练循环和模型保存逻辑
"""

import os
import torch
import torch.nn as nn
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Optional
import time
from tqdm import tqdm

# 导入模型组件
from model import Config, TransformerLanguageModel, generate_text

warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

class TextDataset(Dataset):
    """文本数据集类
    
    将长文本切分成固定长度的序列，用于语言建模训练
    """
    def __init__(self, data: list, tokenizer, max_length: int):
        """
        Args:
            data: token化后的数据列表
            tokenizer: 分词器
            max_length: 序列最大长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data) - self.max_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个训练样本
        
        Returns:
            x: 输入序列 (max_length,)
            y: 目标序列 (max_length,) - 输入序列向右偏移一位
        """
        chunk = self.data[idx:idx + self.max_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def load_data(data_file: str = '../data/scifi.txt') -> Tuple[Optional[list], Optional[list], Optional[object]]:
    """加载和预处理文本数据
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        train_data: 训练数据token列表
        val_data: 验证数据token列表 
        tokenizer: 分词器对象
    """
    print("=" * 60)
    print("📚 数据加载和预处理")
    print("=" * 60)
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ 错误：找不到数据文件 '{data_file}'")
        print("请确保数据文件在指定路径中")
        return None, None, None
    
    # 读取文本数据
    print(f"正在读取文件: {data_file}")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None, None, None
    
    print(f"✅ 数据加载完成")
    print(f"   - 总字符数: {len(text):,}")
    print(f"   - 预览: {text[:100]}...")
    
    # 使用GPT-3兼容的tokenizer
    print("\n正在进行Token化...")
    try:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = tokenizer.encode(text)
    except Exception as e:
        print(f"❌ Token化失败: {e}")
        return None, None, None
    
    print(f"✅ Token化完成")
    print(f"   - 总token数: {len(tokens):,}")
    print(f"   - 词汇表大小: {tokenizer.n_vocab:,}")
    print(f"   - 压缩比: {len(text)/len(tokens):.2f} 字符/token")
    
    # 数据分割：90%训练，10%验证
    split_idx = int(0.9 * len(tokens))
    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]
    
    print(f"\n📊 数据分割:")
    print(f"   - 训练数据: {len(train_data):,} tokens ({len(train_data)/len(tokens)*100:.1f}%)")
    print(f"   - 验证数据: {len(val_data):,} tokens ({len(val_data)/len(tokens)*100:.1f}%)")
    
    return train_data, val_data, tokenizer

@torch.no_grad()
def estimate_loss(model: TransformerLanguageModel, train_loader: DataLoader, 
                 val_loader: DataLoader, eval_iters: int, device: str) -> dict:
    """估算训练和验证损失
    
    Args:
        model: 要评估的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        eval_iters: 评估迭代次数
        device: 计算设备
    
    Returns:
        包含训练和验证损失的字典
    """
    out = {}
    model.eval()
    
    for split, dataloader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        
        # 添加评估进度条
        desc = f"评估{split}损失"
        eval_pbar = tqdm(enumerate(dataloader), desc=desc, total=eval_iters, 
                        leave=False, ncols=100, unit="batch")
        
        for k, (xb, yb) in eval_pbar:
            if k >= eval_iters:
                break
            
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
            
            # 更新进度条
            eval_pbar.set_postfix({'损失': f'{loss.item():.4f}'})
        
        eval_pbar.close()
        out[split] = losses.mean()
    
    model.train()
    return out

def create_data_loaders(train_data: list, val_data: list, tokenizer, config: Config) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        tokenizer: 分词器
        config: 配置对象
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    print("\n🔄 创建数据加载器...")
    
    # 创建数据集
    train_dataset = TextDataset(train_data, tokenizer, config.context_length)
    val_dataset = TextDataset(val_data, tokenizer, config.context_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,  # 避免多进程问题
        pin_memory=True if config.device != 'cpu' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device != 'cpu' else False
    )
    
    print(f"✅ 数据加载器创建完成")
    print(f"   - 训练批次数: {len(train_loader)}")
    print(f"   - 验证批次数: {len(val_loader)}")
    print(f"   - 批次大小: {config.batch_size}")
    print(f"   - 序列长度: {config.context_length}")
    
    return train_loader, val_loader

def train_model(config: Config, data_file: str = '../data/scifi.txt', 
               save_path: str = '../model/transformer_model.pth') -> Tuple[Optional[TransformerLanguageModel], Optional[object]]:
    """训练模型主函数
    
    Args:
        config: 模型配置
        data_file: 数据文件路径
        save_path: 模型保存路径
    
    Returns:
        model: 训练好的模型
        tokenizer: 分词器
    """
    print("=" * 60)
    print("🚀 开始训练Transformer语言模型")
    print("=" * 60)
    
    # 加载数据
    train_data, val_data, tokenizer = load_data(data_file)
    if train_data is None:
        print("❌ 数据加载失败，训练中止")
        return None, None
    
    vocab_size = tokenizer.n_vocab
    
    # 显示训练配置
    print(f"\n⚙️ 训练配置:")
    print(f"   - 设备: {config.device}")
    print(f"   - 词汇表大小: {vocab_size:,}")
    print(f"   - 模型维度: {config.d_model}")
    print(f"   - 注意力头数: {config.num_heads}")
    print(f"   - Transformer层数: {config.num_blocks}")
    print(f"   - 上下文长度: {config.context_length}")
    print(f"   - 批次大小: {config.batch_size}")
    print(f"   - 学习率: {config.learning_rate}")
    print(f"   - 训练步数: {config.max_iters}")
    print(f"   - Dropout: {config.dropout}")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(train_data, val_data, tokenizer, config)
    
    # 创建模型
    print(f"\n🏗️ 创建模型...")
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        context_length=config.context_length,
        dropout=config.dropout
    ).to(config.device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=1e-4  # 添加权重衰减
    )
    
    # 学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_iters, eta_min=config.learning_rate * 0.1
    )
    
    # 训练循环
    print(f"\n🔥 开始训练 (设备: {config.device})...")
    print("-" * 60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    # 获取数据迭代器
    train_iter = iter(train_loader)
    
    # 创建进度条
    pbar = tqdm(range(config.max_iters), desc="训练进度", unit="step", ncols=120)
    
    for step in pbar:
        # 评估模型
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, config.eval_iters, config.device)
            
            elapsed_time = time.time() - start_time
            print(f"\n步数 {step:5d} | 训练损失: {losses['train']:.4f} | 验证损失: {losses['val']:.4f} | "
                  f"时间: {elapsed_time:.1f}s | 学习率: {optimizer.param_groups[0]['lr']:.2e}")
            
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())
            
            # 保存最佳模型
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                best_model_path = save_path.replace('.pth', '_best.pth')
                # 确保模型目录存在
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                model.save_pretrained(best_model_path, config)
        
        # 获取一个批次的数据
        try:
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
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 更新进度条描述
        if step % 10 == 0:  # 每10步更新一次进度条，避免过于频繁
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"训练进度 | 损失: {loss.item():.4f} | 学习率: {current_lr:.2e}")
    
    pbar.close()
    
    total_time = time.time() - start_time
    print(f"\n✅ 训练完成！总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 绘制损失曲线
    plot_training_curves(train_losses, val_losses, config)
    
    # 保存最终模型，确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_pretrained(save_path, config)
    
    return model, tokenizer

def plot_training_curves(train_losses: list, val_losses: list, config: Config):
    """绘制训练损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        config: 配置对象
    """
    print("\n📊 绘制训练曲线...")
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    steps = [i * config.eval_interval for i in range(len(train_losses))]
    plt.plot(steps, train_losses, label='Training loss', marker='o', linewidth=2)
    plt.plot(steps, val_losses, label='Validation loss', marker='s', linewidth=2)
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Loss during training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失对数图
    plt.subplot(1, 2, 2)
    plt.semilogy(steps, train_losses, label='Training loss', marker='o', linewidth=2)
    plt.semilogy(steps, val_losses, label='Validation loss', marker='s', linewidth=2)
    plt.xlabel('Training steps')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss during training (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 确保results目录存在
    os.makedirs('../results', exist_ok=True)
    
    # 保存图片到results文件夹
    loss_plot_path = '../results/training_loss.png'
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存为 '{loss_plot_path}'")
    
    # 显示图片（如果在支持的环境中）
    try:
        plt.show()
    except:
        print("注意: 无法显示图片，请查看保存的图片文件")

def demo_generation(model: TransformerLanguageModel, tokenizer, config: Config):
    """演示文本生成
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        config: 配置对象
    """
    print("\n" + "=" * 60)
    print("🎲 文本生成演示")
    print("=" * 60)
    
    model.eval()
    
    # 针对科幻数据的提示词
    prompts = [
        "",  # 无提示
        "The spaceship",
        "In the future",
        "The alien",
        "Technology",
        "The planet",
        "Captain"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. 提示: '{prompt}' {'(无提示)' if not prompt else ''}")
        print("-" * 50)
        
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer, 
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8,
                device=config.device
            )
            
            # 只显示新生成的部分（去掉原始提示）
            if prompt and generated_text.startswith(prompt):
                display_text = generated_text[len(prompt):].strip()
            else:
                display_text = generated_text
            
            print(f"{display_text}")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")

def main():
    """主函数"""
    print("模块化Transformer语言模型 - 训练程序")
    print("=" * 60)
    
    # 创建配置
    config = Config()
    
    # 针对大数据集调整配置
    config.batch_size = 8  # 增加批次大小
    config.context_length = 32  # 增加上下文长度
    config.max_iters = 10000  # 增加训练步数
    config.eval_interval = 1000  # 调整评估间隔
    
    print(f"当前配置:")
    print(f"  设备: {config.device}")
    print(f"  模型参数: d_model={config.d_model}, heads={config.num_heads}, blocks={config.num_blocks}")
    print(f"  训练参数: batch_size={config.batch_size}, max_iters={config.max_iters}")
    print(f"  上下文长度: {config.context_length}")
    
    # 开始训练
    model, tokenizer = train_model(config)
    
    if model is not None and tokenizer is not None:
        # 演示文本生成
        demo_generation(model, tokenizer, config)
        
        print("\n" + "=" * 60)
        print("🎉 训练和演示完成！")
        print("\n生成的文件:")
        print("  - ../model/transformer_model.pth (最终模型)")
        print("  - ../model/transformer_model_best.pth (最佳模型)")
        print("  - ../results/training_loss.png (训练曲线)")
        print("\n使用方法:")
        print("  python example_usage.py  # 交互式文本生成")
        print("=" * 60)
    else:
        print("\n❌ 训练失败")

if __name__ == "__main__":
    main() 