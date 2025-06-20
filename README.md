# 模块化 Transformer 语言模型

基于销售教材数据训练的小型Transformer语言模型，完全模块化实现，适合学习和实验。

## 📁 项目结构

```
21号/
├── model/                      # 🎯 训练好的模型文件
│   ├── transformer_model.pth       # 最终训练模型 (51MB)
│   └── transformer_model_best.pth  # 最佳验证损失模型 (51MB)
├── data/                       # 📚 训练数据
│   ├── sales_textbook.txt          # 销售教材数据 (451KB)
│   ├── scifi.txt                   # 科幻小说数据 (341MB)
│   └── scifi-finetune.json         # 科幻数据JSON格式 (575MB)
├── pretrain/                   # 🚀 预训练相关代码
│   ├── model.py                    # Transformer模型定义
│   ├── transformer_model.py        # 模型架构实现
│   └── train.py                    # 预训练脚本
├── inference/                  # 🎲 推理和测试
│   └── example_usage.py            # 推理演示脚本
├── finetune/                   # 🔧 微调相关代码
│   └── finetune.py                 # 微调脚本
├── results/                    # 📊 训练结果和可视化
│   ├── training_loss.png           # 预训练损失曲线
│   └── finetune_curves.png         # 微调损失曲线
├── requirements.txt            # 📦 项目依赖
└── README.md                   # 📖 项目说明文档
```

##  🚀 快速使用

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 预训练模型
```bash
cd pretrain
python train.py
```

### 3. 微调模型（可选）
```bash
cd finetune
python finetune.py
```

### 4. 使用模型进行推理
```bash
cd inference
python example_usage.py
```

##  📋 详细运行指南

### 🎯 预训练阶段
在 `pretrain/` 目录下进行：
- **模型定义**：`model.py` 和 `transformer_model.py`
- **训练脚本**：`train.py`
- **训练数据**：使用 `data/sales_textbook.txt`

### 🔧 微调阶段  
在 `finetune/` 目录下进行：
- **微调脚本**：`finetune.py`
- **微调数据**：使用 `data/scifi.txt` 或 `data/scifi-finetune.json`

### 🎲 推理测试
在 `inference/` 目录下进行：
- **推理脚本**：`example_usage.py`
- **模型加载**：自动加载 `model/` 目录下的训练好的模型

### 📊 结果查看
在 `results/` 目录下查看：
- **训练曲线**：`training_loss.png`
- **微调曲线**：`finetune_curves.png`

##  📂 各文件夹详细说明

### `model/` - 模型存储
存放训练完成的模型文件：
- 预训练后的模型会自动保存到此目录
- 推理时会从此目录加载模型
- 包含最终模型和最佳验证损失模型

### `data/` - 数据集
存放所有训练数据：
- `sales_textbook.txt`：销售教材文本，用于预训练
- `scifi.txt`：科幻小说文本，用于微调
- `scifi-finetune.json`：JSON格式的微调数据

### `pretrain/` - 预训练模块
包含预训练相关的所有代码：
- `model.py`：模型配置和核心组件
- `transformer_model.py`：完整的Transformer实现
- `train.py`：预训练主脚本

### `inference/` - 推理模块
包含模型使用和测试代码：
- `example_usage.py`：交互式文本生成演示
- 支持多种生成模式和参数调整

### `finetune/` - 微调模块
包含模型微调代码：
- `finetune.py`：在预训练模型基础上进行微调
- 支持自定义数据集和超参数

### `results/` - 结果可视化
存放训练过程的可视化结果：
- 自动生成的损失曲线图
- 帮助分析训练效果和调整参数

##  模型配置

- 参数量：~13M
- 模型维度：64
- 注意力头数：4
- Transformer层数：8
- 上下文长度：16
- 支持设备：CPU / CUDA / MPS (Apple Silicon)

##  模块化架构

### 1. 模型模块 (`model.py`)
```python
from model import Config, TransformerLanguageModel, generate_text

# 创建配置
config = Config()

# 创建模型
model = TransformerLanguageModel(
    vocab_size=50000,
    d_model=config.d_model,
    num_heads=config.num_heads,
    num_blocks=config.num_blocks,
    context_length=config.context_length
)

# 加载预训练模型
model, config = TransformerLanguageModel.from_pretrained('model.pth')
```

### 2. 训练模块 (`train.py`)
```python
from train import train_model
from model import Config

# 自定义配置训练
config = Config()
config.max_iters = 10000  # 更多训练步数
model, tokenizer = train_model(config)
```

### 3. 推理模块 (`example_usage.py`)
- 🎲 交互式文本生成
- 📦 批量生成演示  
- ⚡ 性能基准测试
- 🎯 多种生成模式

##  使用说明

1. 确保 `sales_textbook.txt` 在项目目录中
2. 运行 `python train.py` 进行训练
3. 运行 `python example_usage.py` 进行文本生成

## 💻 运行命令详解

### 预训练命令
```bash
# 基础预训练
cd pretrain
python train.py

# 自定义训练参数
python train.py --max_iters 10000 --batch_size 32

# 指定GPU设备
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 微调命令
```bash
# 基础微调（需要先有预训练模型）
cd finetune
python finetune.py

# 指定模型路径微调
python finetune.py --model_path ../model/transformer_model_best.pth

# 自定义微调数据
python finetune.py --data_path ../data/custom_data.txt
```

### 推理命令
```bash
# 交互式推理
cd inference
python example_usage.py

# 使用特定模型文件
python example_usage.py --model_path ../model/transformer_model_best.pth

# 批量生成文本
python example_usage.py --batch_mode --num_samples 10
```

### 其他实用命令
```bash
# 查看模型大小
ls -lh model/

# 查看训练日志
tail -f pretrain/training.log

# 清理缓存文件
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### API使用示例
```python
from model import TransformerLanguageModel, generate_text
import tiktoken

# 加载模型
model, config = TransformerLanguageModel.from_pretrained('transformer_model.pth')
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

# 生成文本
text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Sales techniques",
    max_new_tokens=100,
    temperature=0.8,
    device=config.device
)
print(text)
```

##  功能特点

- **🔧 完全模块化**：模型、训练、推理分离
- **🚀 MPS支持**：Apple Silicon Mac GPU加速
- **🎲 交互式生成**：支持带提示的文本生成
- **📊 可视化训练**：自动生成损失曲线
- **⚡ 性能测试**：内置生成速度基准
- **💾 模型管理**：便捷的保存和加载接口

## ⚠️ 注意事项

### 运行顺序
1. **首先预训练**：必须先运行 `pretrain/train.py` 生成基础模型
2. **然后微调**：可选步骤，使用 `finetune/finetune.py` 进行领域适应
3. **最后推理**：使用 `inference/example_usage.py` 生成文本

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA (可选，GPU加速)
- 至少8GB内存（大数据集需要更多）

### 常见问题

**Q: 训练时出现内存不足？**
```bash
# 减少批次大小
python train.py --batch_size 16

# 或使用梯度累积
python train.py --gradient_accumulation_steps 4
```

**Q: 模型文件找不到？**
```bash
# 确保在正确目录运行
pwd  # 应该显示包含model/文件夹的路径
ls model/  # 确认模型文件存在
```

**Q: 推理速度慢？**
```bash
# 使用GPU加速
export CUDA_VISIBLE_DEVICES=0
python example_usage.py

# 或减少生成长度
python example_usage.py --max_new_tokens 50
```

## 🛠️ 开发和扩展

### 自定义模型配置
```python
# 在pretrain/model.py中修改Config类
from model import Config

config = Config()
config.d_model = 128      # 更大的模型
config.num_heads = 8      # 更多注意力头
config.num_blocks = 12    # 更深的网络
config.max_iters = 10000  # 更长训练
```

### 添加自定义数据
```bash
# 1. 将新数据放入data/目录
cp your_data.txt data/

# 2. 修改训练脚本使用新数据
cd pretrain
python train.py --data_path ../data/your_data.txt
```

### 自定义生成参数
```python
# 在inference/example_usage.py中调整
temperature = 0.8    # 控制随机性 (0.1-2.0)
top_k = 50          # 候选词数量
max_tokens = 100    # 生成长度
```

### 数据格式要求
支持以下格式：
- **纯文本**：`.txt` 文件，UTF-8编码
- **JSON**：包含 `"text"` 字段的JSON数组
- **自动处理**：90%/10% 训练/验证分割

## 🎯 快速开始总结

```bash
# 1. 克隆/下载项目到本地
# 2. 安装依赖
pip install -r requirements.txt

# 3. 预训练模型（必须步骤）
cd pretrain
python train.py

# 4. 微调模型（可选）
cd ../finetune
python finetune.py

# 5. 测试推理
cd ../inference
python example_usage.py

# 6. 查看结果
cd ../results
# 查看训练曲线图
```