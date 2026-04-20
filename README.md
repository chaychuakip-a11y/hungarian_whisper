# Hungarian Whisper Fine-tuning Pipeline

匈牙利语 Whisper 模型微调流水线，支持 CUDA / ROCm 双平台。

## 目录结构

```
hungarian_whisper/
├── config/                     # 配置文件
│   ├── config.yaml            # CUDA/NVIDIA 配置
│   ├── config_rocm.yaml       # ROCm/AMD 配置
│   └── ctc_phone_hu.yaml     # 匈牙利语音素配置
├── src/                        # 源代码 (HuggingFace Trainer)
│   ├── data/                  # 数据处理
│   ├── model/                 # 模型
│   ├── training/              # 训练
│   └── utils/                 # 工具
├── scripts/
│   ├── train_hulk.sh          # Hulk 框架训练配置
│   ├── prepare_hulk_data.py   # Hulk 数据准备
│   ├── train_rocm_simple.py   # ROCm 快速训练
│   └── ...
├── data/                      # 数据目录
├── output/                    # 输出目录
└── Hungarian_Whisper_Tuning_Report.md
```

## 三种训练方式

### 1. HuggingFace Trainer (已验证 ROCm)
```bash
python scripts/train_rocm_simple.py   # 快速验证
```

### 2. CUDA 训练
```bash
bash scripts/03_train.sh
```

### 3. Hulk 框架训练 (基于公司框架)
```bash
# 准备数据
python scripts/prepare_hulk_data.py --output_dir ./data/hulk_lmdb --generate_synthetic

# 生成配置
bash scripts/train_hulk.sh

# 运行训练 (需要 Hulk 框架环境)
torchrun ... $HULK_DIR/hulk_cli/hydra_train.py --config-dir ./output/hulk_hungarian ...
```

## Hulk 框架配置

### YAML 配置示例
```yaml
model:
  _name: model_parallel_whisper_distil_dec
  from_pretrained: /path/to/pretrained.pt

  n_audio_ctx: 1500
  n_audio_state: 1280
  n_text_state: 1280
  n_audio_layer: 32
  n_text_layer: 4

lora:
  apply_lora: true
  lora_rank: 16
  lora_alpha: 128
  lora_dropout: 0.05
  adapt_q: true
  adapt_k: true
  adapt_v: true
  adapt_o: true

task:
  _name: multilingual_asr_task
  feature_type: "fb80"
  chunk_seq_path: /path/to/chunk.bin

dataset:
  train_subset: /path/to/train.json
  batch_size: 512
  max_tokens: 4096
```

## 数据格式

### Hulk LMDB 格式
```
data/hulk_lmdb/
├── hungarian_lmdb/           # LMDB 数据库
├── hungarian_chunk10000.bin  # Chunk 索引
├── hungarian_dataset.json    # 数据集描述
└── phone_dict.json          # 音素字典
```

### HTK 格式
```
data/htk_output/
├── wav.scp                  # 音频路径列表
└── labels.mlf              # Master Label File
```

## 匈牙利语处理

### 音素集合 (48 phones)
```python
HUNGARIAN_PHONES = [
    '<SIL>', '<SP>',
    'a', 'e', 'i', 'o', 'u', 'y',          # Vowels
    'á', 'é', 'í', 'ó', 'ö', 'ő', 'ú', 'ü', 'ű',  # Long vowels
    'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
    'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z',  # Consonants
    'cs', 'dz', 'dzs', 'gy', 'ly', 'ny', 'sz', 'ty', 'zs'  # Digraphs
]
```

## 平台支持

| 组件 | CUDA (NVIDIA) | ROCm (AMD) |
|------|---------------|-------------|
| PyTorch | ✅ torch (CUDA) | ✅ torch+rocm5.7 |
| INT8 量化 | ✅ bitsandbytes | ❌ 不支持 |
| BF16/FP16 | ✅ | ✅ |
| LoRA | ✅ | ✅ |
| Hulk 框架 | ✅ | ✅ |

### ROCm 验证结果
- GPU: AMD Radeon RX 7900 XTX (25.75GB)
- VRAM 使用: ~3.4GB
- 状态: ✅ 已验证

## 快速开始

### ROCm (推荐)
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
pip install transformers datasets peft accelerate librosa evaluate scipy soundfile pyyaml jiwer tensorboard

python scripts/train_rocm_simple.py
```

### CUDA
```bash
bash scripts/01_install_deps.sh
bash scripts/02_prepare_data.sh
bash scripts/03_train.sh
```

### Hulk 框架 (需要公司环境)
```bash
python scripts/prepare_hulk_data.py --output_dir ./data/hulk_lmdb --generate_synthetic
bash scripts/train_hulk.sh
```

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
