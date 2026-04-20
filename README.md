# Hungarian Whisper Fine-tuning Pipeline

匈牙利语 Whisper 模型微调流水线，支持 CUDA 和 ROCm 双平台。

## 目录结构

```
hungarian_whisper/
├── config/                     # 配置文件
│   ├── config.yaml            # CUDA/NVIDIA 配置
│   ├── config_rocm.yaml       # ROCm/AMD 配置
│   └── ctc_phone_hu.yaml     # 匈牙利语音素配置
├── src/
│   ├── data/                  # 数据处理
│   │   ├── hungarian_normalizer.py   # 匈牙利语文本标准化
│   │   ├── htk_exporter.py           # HTK 格式导出
│   │   ├── htk_dataloader.py         # HTK 数据加载
│   │   ├── dataset_loader.py         # HuggingFace 数据集加载
│   │   ├── lmdb_preparator.py       # LMDB 格式准备
│   │   └── collator.py              # 数据整理器
│   ├── model/
│   │   └── lora_whisper.py          # LoRA 模型配置
│   ├── training/
│   │   ├── trainer.py               # 训练循环
│   │   └── evaluation.py           # WER 评估
│   └── utils/
│       ├── lora_layers.py          # LoRA 层实现
│       └── memory_monitor.py        # VRAM 监控
├── scripts/
│   ├── 01_install_deps.sh          # 安装依赖
│   ├── 02_prepare_data.sh          # 数据准备
│   ├── 03_train.sh                 # CUDA 训练
│   ├── 03_train_rocm.sh            # ROCm 训练
│   ├── 04_evaluate.sh              # 评估
│   ├── run_pipeline.py             # 一键运行完整流程
│   ├── train_rocm_simple.py        # ROCm 简化训练
│   └── train_rocm_full.py          # ROCm 完整训练
├── data/                          # 数据目录
├── output/                        # 输出目录
└── Hungarian_Whisper_Tuning_Report.md
```

## 快速开始

### 1. 安装依赖

**CUDA (NVIDIA):**
```bash
bash scripts/01_install_deps.sh
```

**ROCm (AMD):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
pip install transformers datasets peft accelerate librosa evaluate scipy soundfile pyyaml jiwer tensorboard
```

### 2. 数据准备

```bash
# 生成合成数据测试
python scripts/test_pipeline.py --num_samples 500

# 或准备真实数据
bash scripts/02_prepare_data.sh
```

### 3. 训练

**CUDA 训练:**
```bash
bash scripts/03_train.sh
```

**ROCm 训练 (推荐):**
```bash
python scripts/train_rocm_simple.py   # 快速验证
python scripts/train_rocm_full.py     # 完整 LoRA 训练
```

### 4. 评估

```bash
bash scripts/04_evaluate.sh
```

## 配置文件

### config.yaml (CUDA)
```yaml
model:
  name: "openai/whisper-large-v2"
  int8: true                    # NVIDIA 支持 INT8
  lora:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  bf16: true
```

### config_rocm.yaml (ROCm)
```yaml
model:
  name: "openai/whisper-large-v2"
  int8: false                   # AMD 不支持 INT8
  lora:
    r: 16
    lora_alpha: 32

training:
  per_device_train_batch_size: 2  # 降低 batch size
  bf16: true
```

## 数据格式

### HTK 格式

**wav.scp:**
```
sample_000001 /path/to/audio_001.wav
sample_000002 /path/to/audio_002.wav
```

**labels.mlf:**
```shell
#!MLF!#
"*/sample_000001.lab"
köszönöm
szépen
üdvözöllek
```

## ROCm 支持

| 组件 | CUDA (NVIDIA) | ROCm (AMD) |
|------|---------------|------------|
| PyTorch | torch (CUDA) | torch+rocm5.7 |
| INT8 量化 | bitsandbytes | 不支持 |
| BF16/FP16 | ✅ | ✅ |
| LoRA | ✅ | ✅ |

### ROCm 验证结果
- GPU: AMD Radeon RX 7900 XTX (25.75GB)
- 模型: whisper-small
- VRAM 使用: ~3.4GB
- 状态: ✅ 已验证

## 匈牙利语处理

### 有效字符
```
aáäbcdeééfghiíjkloóöőpqrstuúüűvwxyz
AÁÄBCDEÉÉFGHIÍJKLOÓÖŐPQRSTUÚÜŰVWXYZ
```

### 文本标准化规则
1. 缩写扩展: "kb." → "körülbelül"
2. 数字转换: "123" → "százhuszonhárom"
3. 标点过滤: 仅保留 . , ? ! : - ' " ( )
4. 字符过滤: 移除非匈牙利语字符

## 依赖

```
transformers==4.36.0
datasets==2.16.0
peft==0.7.0
accelerate==0.25.0
librosa==0.10.1
evaluate==0.4.1
scipy==1.11.4
soundfile==0.12.1
pyyaml==6.0.1
jiwer
tensorboard
```

## License

MIT
