#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化Transformer语言模型 - 推理演示脚本

提供交互式文本生成和批量生成功能
"""

import os
import torch
import tiktoken
import sys
from typing import Optional

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型组件
from pretrain.model import TransformerLanguageModel, Config, generate_text

def load_model(model_path: str = '../model/transformer_model.pth') -> tuple[Optional[TransformerLanguageModel], Optional[object], Optional[Config]]:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        model: 加载的模型
        tokenizer: 分词器
        config: 配置对象
    """
    print("=" * 60)
    print("🔄 加载模型")
    print("=" * 60)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 '{model_path}'")
        print("请先运行 'python train.py' 训练模型")
        return None, None, None
    
    try:
        # 加载模型
        print(f"正在加载模型: {model_path}")
        model, config = TransformerLanguageModel.from_pretrained(model_path)
        
        # 获取tokenizer
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        print(f"✅ 模型加载成功")
        print(f"   - 模型参数: {model.get_num_params()/1e6:.2f}M")
        print(f"   - 模型维度: {config.d_model}")
        print(f"   - 注意力头数: {config.num_heads}")
        print(f"   - Transformer层数: {config.num_blocks}")
        print(f"   - 上下文长度: {config.context_length}")
        print(f"   - 设备: {config.device}")
        
        return model, tokenizer, config
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None

def interactive_generation(model: TransformerLanguageModel, tokenizer, config: Config):
    """交互式文本生成
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        config: 配置对象
    """
    print("\n" + "=" * 60)
    print("🎲 交互式文本生成")
    print("=" * 60)
    print("输入提示文本，模型将生成续写内容")
    print("命令:")
    print("  - 直接输入文本作为提示")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'random' 进行随机生成（无提示）")
    print("  - 输入 'batch' 进行批量演示")
    print("  - 输入 'help' 查看帮助")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\n📝 输入提示 (或命令): ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            elif prompt.lower() == 'help':
                print("\n帮助信息:")
                print("  - 输入任何文本作为生成提示")
                print("  - 'random': 随机生成")
                print("  - 'batch': 批量演示")
                print("  - 'quit'/'exit': 退出程序")
                continue
            elif prompt.lower() == 'batch':
                batch_generation_demo(model, tokenizer, config)
                continue
            elif prompt.lower() == 'random':
                prompt = ""
                print("🎲 随机生成...")
            
            # 设置生成参数
            max_tokens = 100
            temperature = 0.8
            
            print(f"\n⚙️ 生成参数: max_tokens={max_tokens}, temperature={temperature}")
            print("🚀 生成中...")
            print("-" * 50)
            
            # 生成文本
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                device=config.device
            )
            
            # 显示结果
            if prompt:
                print(f"📥 原始提示: {prompt}")
                if generated_text.startswith(prompt):
                    new_text = generated_text[len(prompt):].strip()
                    print(f"✨ 生成内容: {new_text}")
                else:
                    print(f"✨ 完整生成: {generated_text}")
            else:
                print(f"✨ 随机生成: {generated_text}")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")

def batch_generation_demo(model: TransformerLanguageModel, tokenizer, config: Config):
    """批量生成演示
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        config: 配置对象
    """
    print("\n" + "=" * 60)
    print("📦 批量生成演示")
    print("=" * 60)
    
    # 预定义的提示词
    prompts = [
        "",  # 随机生成
        "Sales",
        "Building rapport",
        "Customer service",
        "Communication skills",
        "The importance of",
        "In business",
        "Effective communication",
        "Understanding customers",
        "Building relationships"
    ]
    
    # 不同的温度设置
    temperatures = [0.5, 0.8, 1.2]
    
    print(f"将为 {len(prompts)} 个提示各生成 {len(temperatures)} 个版本")
    print("使用不同的温度参数控制创造性\n")
    
    for i, prompt in enumerate(prompts, 1):
        prompt_display = f"'{prompt}'" if prompt else "'随机生成'"
        print(f"\n{i}. 提示: {prompt_display}")
        print("=" * 50)
        
        for j, temp in enumerate(temperatures, 1):
            print(f"\n  📊 版本 {j} (温度={temp}):")
            print("  " + "-" * 45)
            
            try:
                generated_text = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=50,  # 较短的生成用于演示
                    temperature=temp,
                    device=config.device
                )
                
                # 处理显示文本
                if prompt and generated_text.startswith(prompt):
                    display_text = generated_text[len(prompt):].strip()
                else:
                    display_text = generated_text
                
                # 格式化输出
                words = display_text.split()
                lines = []
                current_line = "  "
                
                for word in words:
                    if len(current_line + word) > 70:  # 控制行长度
                        lines.append(current_line.rstrip())
                        current_line = "  " + word + " "
                    else:
                        current_line += word + " "
                
                if current_line.strip():
                    lines.append(current_line.rstrip())
                
                for line in lines:
                    print(line)
                
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")

def benchmark_generation(model: TransformerLanguageModel, tokenizer, config: Config):
    """生成性能测试
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        config: 配置对象
    """
    print("\n" + "=" * 60)
    print("⚡ 生成性能测试")
    print("=" * 60)
    
    import time
    
    test_prompts = ["Sales", "Customer", "Building"]
    test_lengths = [50, 100, 200]
    
    print(f"测试配置:")
    print(f"  - 提示数量: {len(test_prompts)}")
    print(f"  - 生成长度: {test_lengths}")
    print(f"  - 设备: {config.device}")
    print()
    
    for length in test_lengths:
        print(f"📏 测试长度: {length} tokens")
        total_time = 0
        total_tokens = 0
        
        for prompt in test_prompts:
            start_time = time.time()
            
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=length,
                temperature=1.0,
                device=config.device
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_generated = len(tokenizer.encode(generated_text)) - len(tokenizer.encode(prompt))
            
            total_time += generation_time
            total_tokens += tokens_generated
            
            print(f"  {prompt:15s}: {generation_time:.2f}s, {tokens_generated} tokens, {tokens_generated/generation_time:.1f} tokens/s")
        
        avg_speed = total_tokens / total_time
        print(f"  平均速度: {avg_speed:.1f} tokens/s\n")

def main():
    """主函数"""
    print("模块化Transformer语言模型 - 推理演示")
    print("版本: 2.0 (模块化)")
    
    # 检查命令行参数
    model_path = '../model/transformer_model.pth'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # 加载模型
    model, tokenizer, config = load_model(model_path)
    
    if model is None:
        print("\n❌ 无法加载模型，程序退出")
        return
    
    print("\n🎯 选择操作模式:")
    print("1. 交互式生成 (默认)")
    print("2. 批量演示")
    print("3. 性能测试")
    print("4. 全部运行")
    
    try:
        choice = input("\n选择模式 (1-4, 默认为1): ").strip()
        
        if choice == '2':
            batch_generation_demo(model, tokenizer, config)
        elif choice == '3':
            benchmark_generation(model, tokenizer, config)
        elif choice == '4':
            batch_generation_demo(model, tokenizer, config)
            benchmark_generation(model, tokenizer, config)
            interactive_generation(model, tokenizer, config)
        else:  # 默认为交互式
            interactive_generation(model, tokenizer, config)
            
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 感谢使用模块化Transformer语言模型！")
    print("=" * 60)

if __name__ == "__main__":
    main() 