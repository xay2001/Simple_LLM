#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer语言模型微调脚本

使用科幻小说数据集对预训练的Transformer模型进行微调
"""

import os
import sys
import json
import torch
import torch.nn as nn
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Optional, List, Dict
import time
from tqdm import tqdm
import argparse

# 添加父目录到路径以便导入模型
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pretrain'))

# 导入模型组件
from model import Config, TransformerLanguageModel, generate_text

warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

class FinetuneConfig(Config):
    """微调专用配置类，继承预训练配置"""
    def __init__(self):
        super().__init__()
        # 微调专用参数
        self.max_iters = 1000
        self.learning_rate = 5e-5  # 微调通常使用更小的学习率
        self.eval_interval = 100
        self.eval_iters = 50
        self.warmup_steps = 100
        self.weight_decay = 0.01
        self.save_interval = 200
        
        # 数据路径
        self.pretrained_model_path = '../model/transformer_model.pth'
        self.finetune_data_path = '../data/scifi-finetune.json'
        self.output_model_path = '../model/transformer_model_finetuned.pth'
        
        print(f"🔧 微调配置:")
        print(f"   - 设备: {self.device}")
        print(f"   - 批次大小: {self.batch_size}")
        print(f"   - 上下文长度: {self.context_length}")
        print(f"   - 学习率: {self.learning_rate}")
        print(f"   - 最大迭代: {self.max_iters}")

class FinetuneDataset(Dataset):
    """微调数据集类
    
    处理instruction-input-output格式的数据
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int):
        """
        Args:
            data: 包含instruction, input, output的字典列表
            tokenizer: 分词器
            max_length: 序列最大长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._process_data()
        
    def _process_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """预处理数据，将文本转换为token序列"""
        processed = []
        
        print("🔄 预处理微调数据...")
        for item in tqdm(self.data, desc="处理数据", ncols=80, leave=False):
            # 构造训练文本格式
            if item.get('input', '').strip():
                # 有input的情况
                text = f"指令：{item['instruction']}\n输入：{item['input']}\n输出：{item['output']}"
            else:
                # 没有input的情况
                text = f"指令：{item['instruction']}\n输出：{item['output']}"
            
            # 对文本进行tokenization
            tokens = self.tokenizer.encode(text)
            
            # 如果序列太短，跳过
            if len(tokens) < 2:
                continue
                
            # 如果序列太长，截断
            if len(tokens) > self.max_length + 1:
                tokens = tokens[:self.max_length + 1]
            
            # 创建输入和目标序列
            if len(tokens) > 1:
                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]
                
                # 填充到固定长度
                while len(input_tokens) < self.max_length:
                    input_tokens.append(self.tokenizer.eot_token)  # 使用结束token填充
                    target_tokens.append(self.tokenizer.eot_token)
                
                x = torch.tensor(input_tokens[:self.max_length], dtype=torch.long)
                y = torch.tensor(target_tokens[:self.max_length], dtype=torch.long)
                processed.append((x, y))
        
        print(f"✅ 数据预处理完成，有效样本数: {len(processed)}")
        return processed
        
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.processed_data[idx]

def load_finetune_data(data_path: str) -> Tuple[Optional[List[Dict]], Optional[object]]:
    """加载微调数据
    
    Args:
        data_path: JSON数据文件路径
    
    Returns:
        data: 数据列表
        tokenizer: 分词器对象
    """
    print("=" * 60)
    print("📚 加载微调数据")
    print("=" * 60)
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到数据文件 '{data_path}'")
        return None, None
    
    # 读取JSON数据
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取JSON文件失败: {e}")
        return None, None
    
    # 验证数据格式
    if not isinstance(data, list):
        print("❌ 数据格式错误：应该是包含字典的列表")
        return None, None
    
    # 检查数据条目格式
    valid_data = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"⚠️ 跳过第{i+1}条数据：不是字典格式")
            continue
        
        if 'instruction' not in item or 'output' not in item:
            print(f"⚠️ 跳过第{i+1}条数据：缺少必要字段")
            continue
            
        valid_data.append(item)
    
    print(f"✅ 数据加载完成")
    print(f"   - 总条目数: {len(data)}")
    print(f"   - 有效条目数: {len(valid_data)}")
    
    if len(valid_data) > 0:
        # 显示数据样例
        sample = valid_data[0]
        print(f"   - 数据样例:")
        print(f"     指令: {sample['instruction'][:50]}...")
        if sample.get('input'):
            print(f"     输入: {sample['input'][:50]}...")
        print(f"     输出: {sample['output'][:50]}...")
    
    # 初始化tokenizer
    try:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        print(f"   - 词汇表大小: {tokenizer.n_vocab:,}")
    except Exception as e:
        print(f"❌ Tokenizer初始化失败: {e}")
        return None, None
    
    return valid_data, tokenizer

def load_pretrained_model(model_path: str, tokenizer, config: FinetuneConfig) -> Optional[TransformerLanguageModel]:
    """加载预训练模型
    
    Args:
        model_path: 模型文件路径
        tokenizer: 分词器
        config: 配置对象
    
    Returns:
        加载的模型
    """
    print("\n🔄 加载预训练模型...")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到预训练模型文件 '{model_path}'")
        return None
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=config.device)
        
        # 检查checkpoint格式并提取信息
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整的checkpoint格式
            state_dict = checkpoint['model_state_dict']
            saved_config = checkpoint.get('config', {})
            saved_vocab_size = checkpoint.get('vocab_size', tokenizer.n_vocab)
            
            print(f"✅ 检测到完整checkpoint格式")
            print(f"   - 保存的词汇表大小: {saved_vocab_size}")
            print(f"   - 保存的配置: {saved_config}")
            
            # 更新配置以匹配预训练模型
            if 'd_model' in saved_config:
                config.d_model = saved_config['d_model']
            if 'num_heads' in saved_config:
                config.num_heads = saved_config['num_heads']
            if 'num_blocks' in saved_config:
                config.num_blocks = saved_config['num_blocks']
            if 'context_length' in saved_config:
                config.context_length = saved_config['context_length']
            if 'dropout' in saved_config:
                config.dropout = saved_config['dropout']
                
            # 使用保存的词汇表大小
            vocab_size = saved_vocab_size
            
        else:
            # 直接的状态字典格式
            state_dict = checkpoint
            vocab_size = tokenizer.n_vocab
            print("✅ 检测到状态字典格式")
        
        # 创建模型实例
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_blocks=config.num_blocks,
            context_length=config.context_length,
            dropout=config.dropout
        )
        
        # 加载预训练权重
        model.load_state_dict(state_dict)
        model.to(config.device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 预训练模型加载完成")
        print(f"   - 模型参数总数: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

@torch.no_grad()
def estimate_loss(model: TransformerLanguageModel, train_loader: DataLoader, 
                 val_loader: DataLoader, eval_iters: int, device: str) -> dict:
    """估算训练和验证损失"""
    out = {}
    model.eval()
    
    for split, dataloader in [('train', train_loader), ('val', val_loader)]:
        if dataloader is None:
            out[split] = float('inf')
            continue
            
        losses = []
        count = 0
        
        for xb, yb in dataloader:
            if count >= eval_iters:
                break
            
            xb, yb = xb.to(device), yb.to(device)
            try:
                logits, loss = model(xb, yb)
                losses.append(loss.item())
                count += 1
            except Exception as e:
                print(f"⚠️ 评估时出现错误: {e}")
                continue
        
        if losses:
            out[split] = np.mean(losses)
        else:
            out[split] = float('inf')
    
    model.train()
    return out

def create_optimizer(model: TransformerLanguageModel, config: FinetuneConfig):
    """创建优化器和学习率调度器"""
    
    # 为不同的参数组设置不同的权重衰减
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name and 'embedding' not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
    
    optimizer_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_groups, lr=config.learning_rate)
    
    # 创建学习率调度器
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (config.max_iters - config.warmup_steps)
            return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def train_model(config: FinetuneConfig) -> Tuple[Optional[TransformerLanguageModel], Optional[object]]:
    """微调模型主函数"""
    
    print("=" * 60)
    print("🚀 开始微调Transformer语言模型")
    print("=" * 60)
    
    # 1. 加载数据
    data, tokenizer = load_finetune_data(config.finetune_data_path)
    if data is None or tokenizer is None:
        return None, None
    
    # 2. 加载预训练模型
    model = load_pretrained_model(config.pretrained_model_path, tokenizer, config)
    if model is None:
        return None, None
    
    # 3. 数据分割
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:] if len(data) > split_idx else None
    
    print(f"\n📊 数据分割:")
    print(f"   - 训练样本: {len(train_data)}")
    print(f"   - 验证样本: {len(val_data) if val_data else 0}")
    
    # 4. 创建数据集和数据加载器
    train_dataset = FinetuneDataset(train_data, tokenizer, config.context_length)
    val_dataset = FinetuneDataset(val_data, tokenizer, config.context_length) if val_data else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device != 'cpu' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device != 'cpu' else False
    ) if val_dataset else None
    
    # 5. 创建优化器
    optimizer, scheduler = create_optimizer(model, config)
    
    # 6. 训练循环
    print("\n🔥 开始训练...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 初始评估
    print("\n📊 初始模型评估...")
    losses = estimate_loss(model, train_loader, val_loader, config.eval_iters, config.device)
    print(f"初始损失 - 训练: {losses['train']:.4f}, 验证: {losses['val']:.4f}")
    
    # 训练进度条
    progress_bar = tqdm(range(config.max_iters), desc="微调进度", 
                       ncols=100, dynamic_ncols=True, leave=True)
    
    model.train()
    train_iter = iter(train_loader)
    
    for iter_num in progress_bar:
        try:
            # 获取批次数据
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)  # 重新开始
                xb, yb = next(train_iter)
            
            xb, yb = xb.to(config.device), yb.to(config.device)
            
            # 前向传播
            logits, loss = model(xb, yb)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            train_losses.append(loss.item())
            
            # 更新进度条
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                '损失': f'{loss.item():.4f}',
                '学习率': f'{current_lr:.2e}'
            })
            
            # 定期评估
            if (iter_num + 1) % config.eval_interval == 0:
                print(f"\n📊 第 {iter_num + 1} 轮评估...")
                losses = estimate_loss(model, train_loader, val_loader, config.eval_iters, config.device)
                val_losses.append(losses['val'])
                
                print(f"损失 - 训练: {losses['train']:.4f}, 验证: {losses['val']:.4f}")
                
                # 更新最佳验证损失（但不保存）
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    print(f"🎯 发现更好模型，验证损失: {best_val_loss:.4f}")
                
                # 生成样例文本
                print("\n📝 生成样例:")
                sample_text = generate_text(
                    model, tokenizer, 
                    prompt="指令：请续写这个科幻故事\n输入：在遥远的未来\n输出：", 
                    max_new_tokens=50,
                    device=config.device
                )
                print(f"   {sample_text}")
            
            # 移除检查点保存，只在最后保存最终模型
                
        except Exception as e:
            print(f"\n⚠️ 训练步骤 {iter_num} 出现错误: {e}")
            continue
    
    progress_bar.close()
    
    # 7. 保存最终模型
    print(f"\n💾 保存最终模型到: {config.output_model_path}")
    
    # 保存完整的模型信息（包括配置）
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': config.d_model,
            'num_heads': config.num_heads,
            'num_blocks': config.num_blocks,
            'context_length': config.context_length,
            'dropout': config.dropout
        },
        'vocab_size': model.vocab_size,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else None
    }, config.output_model_path)
    
    # 8. 绘制训练曲线
    if len(val_losses) > 0:
        plot_training_curves(train_losses, val_losses, config)
    
    print("\n🎉 微调完成!")
    print(f"   - 最佳验证损失: {best_val_loss:.4f}")
    print(f"   - 模型保存路径: {config.output_model_path}")
    
    return model, tokenizer

def plot_training_curves(train_losses: List[float], val_losses: List[float], config: FinetuneConfig):
    """绘制训练曲线"""
    try:
        plt.figure(figsize=(12, 4))
        
        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, alpha=0.7, label='训练损失')
        plt.title('训练损失曲线')
        plt.xlabel('迭代步数')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 验证损失
        plt.subplot(1, 2, 2)
        eval_steps = [(i + 1) * config.eval_interval for i in range(len(val_losses))]
        plt.plot(eval_steps, val_losses, 'o-', label='验证损失', color='orange')
        plt.title('验证损失曲线')
        plt.xlabel('迭代步数')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/finetune_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📈 训练曲线已保存到 ../results/finetune_curves.png")
        
    except Exception as e:
        print(f"⚠️ 绘制训练曲线失败: {e}")

def demo_generation(model: TransformerLanguageModel, tokenizer, config: FinetuneConfig):
    """演示文本生成"""
    print("\n" + "=" * 60)
    print("📝 文本生成演示")
    print("=" * 60)
    
    prompts = [
        "指令：续写科幻小说\n输入：太空中的战舰\n输出：",
        "指令：描述未来世界\n输出：",
        "指令：创作机器人故事\n输入：智能机器人觉醒\n输出：",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🔮 示例 {i}:")
        print(f"输入: {prompt}")
        
        generated = generate_text(
            model, tokenizer, 
            prompt=prompt, 
            max_new_tokens=100,
            device=config.device
        )
        print(f"生成: {generated}")
        print("-" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='微调Transformer语言模型')
    parser.add_argument('--data_path', type=str, default='../data/scifi-finetune.json',
                       help='微调数据路径')
    parser.add_argument('--model_path', type=str, default='../model/transformer_model.pth',
                       help='预训练模型路径')
    parser.add_argument('--output_path', type=str, default='../model/transformer_model_finetuned.pth',
                       help='输出模型路径')
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='最大训练迭代数')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 创建配置
    config = FinetuneConfig()
    config.finetune_data_path = args.data_path
    config.pretrained_model_path = args.model_path
    config.output_model_path = args.output_path
    config.max_iters = args.max_iters
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    
    # 确保输出目录存在
    os.makedirs('../results', exist_ok=True)
    os.makedirs(os.path.dirname(config.output_model_path), exist_ok=True)
    
    # 开始微调
    model, tokenizer = train_model(config)
    
    if model is not None and tokenizer is not None:
        # 演示生成
        demo_generation(model, tokenizer, config)
    else:
        print("❌ 微调失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 