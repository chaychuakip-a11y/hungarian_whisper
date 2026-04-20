# 匈牙利语Whisper模型微调实验报告

## 1. 项目概述

### 1.1 目标
在本地环境中对OpenAI Whisper模型进行匈牙利语微调，使用HTK（Hidden Markov Model Toolkit）标注格式，处理24GB VRAM约束下的训练任务。

### 1.2 关键约束
- **模型**: whisper-large-v2 (1.5B参数) / whisper-small (验证用)
- **量化**: INT8量化 + LoRA微调 (CUDA)
- **VRAM限制**: 24GB
- **数据集**: Common Voice + VoxPopuli + FLEURS（匈牙利语）
- **输出格式**: 训练后的模型 + HTK格式文件

---

## 2. ROCm 平台验证结果

### 2.1 硬件环境
- **GPU**: AMD Radeon RX 7900 XTX
- **VRAM**: 25.75GB
- **ROCm**: 5.7

### 2.2 验证结果
```
============================================================
Hungarian Whisper Training on ROCm (AMD GPU)
============================================================
VRAM: 0.00GB / 25.75GB
[1] Loading Whisper model...
VRAM after model load: 2.46GB
[2] Creating dataset...
[3] Exporting to HTK format...
[4] Starting training...
Step 0/50 | Loss: 2.7171 | VRAM: 3.40GB
Step 10/50 | Loss: 2.57 | VRAM: 3.40GB
Step 20/50 | Loss: 2.63 | VRAM: 3.40GB
Step 30/50 | Loss: 2.60 | VRAM: 3.40GB
Step 40/50 | Loss: 2.59 | VRAM: 3.40GB
Final VRAM: 3.40GB
[5] Saving model...
Model saved to ./output/checkpoints
============================================================
ROCm training completed successfully!
============================================================
```

**结论**: ✅ ROCm 验证通过，VRAM 使用仅 3.4GB/25.75GB

---

## 3. 数据清洗规则

### 3.1 匈牙利语字符集
保留的有效匈牙利语字符：
- 小写: `aáäbcdeééfghiíjkloóöőpqrstuúüűvwxyz`
- 大写: `AÁÄBCDEÉÉFGHIÍJKLOÓÖŐPQRSTUÚÜŰVWXYZ`
- 特殊字符: `á, é, í, ó, ö, ő, ú, ü, ű`

### 3.2 文本标准化流程
```
1. 空白字符规范化: \s+ → 单个空格
2. 缩写扩展:
   - "kb." → "körülbelül"
   - "stb." → "és így tovább"
   - "pl." → "például"
3. 数字转换:
   - 0-20: 直接映射 (0→"nulla", 1→"egy", 等)
   - 21-99: 十位+个位组合
   - 100+: 百位+剩余部分
4. 标点过滤: 仅保留 . , ? ! : - ' " ( )
5. 字符过滤: 移除所有非匈牙利语字符
6. 转小写: 统一为小写格式
```

### 3.3 数据过滤规则
| 条件 | 阈值 |
|------|------|
| 最小音频时长 | 1.0秒 |
| 最大音频时长 | 25.0秒 |
| 最小文本长度 | 2个单词 |
| 禁用纯数字文本 | 是 |

---

## 4. HTK格式转换逻辑

### 4.1 HTK文件格式

#### wav.scp (音频脚本文件)
```
格式: recording_id /full/path/to/audio.wav

示例:
sample_000001 /data/htk_output/audio_000001.wav
sample_000002 /data/htk_output/audio_000002.wav
```

#### labels.mlf (主标签文件)
```shell
#!MLF!#
"*/sample_000001.lab"
szeretnék
üdvözölni
önöket

"*/sample_000002.lab"
köszönöm
szépen
```

### 4.2 转换流程
1. 从HuggingFace加载原始数据集
2. 应用匈牙利语文本标准化
3. 按音频时长过滤 (1-25秒)
4. 验证文本有效性
5. 生成wav.scp文件
6. 生成labels.mlf文件
7. 保存音频文件到统一目录

---

## 5. LoRA配置详情

### 5.1 模型配置
```yaml
model:
  name: "openai/whisper-large-v2"
  int8: true

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]
  lora_dropout: 0.05
```

### 5.2 可训练参数分析
| 参数类别 | 数量 | 占比 |
|----------|------|------|
| 全部参数 | 1.5B | 100% |
| LoRA参数 | ~4M | 0.27% |

**LoRA模块**: q_proj (查询投影), v_proj (值投影)

### 5.3 VRAM估算
| 组件 | 估算VRAM |
|------|----------|
| 模型权重 (INT8) | ~2GB |
| 激活值 (BF16) | ~8GB |
| 梯度 (BF16) | ~2GB |
| 优化器状态 (FP32) | ~8GB |
| **总计** | ~20GB |

---

## 6. 训练配置

### 6.1 ROCm 配置 (config_rocm.yaml)
```yaml
model:
  name: "openai/whisper-small"
  int8: false  # bitsandbytes not supported on ROCm

lora:
  r: 16
  lora_alpha: 128
  target_modules: ["q_proj", "v_proj"]
  lora_dropout: 0.05

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-3
  num_train_epochs: 1
  bf16: true
  warmup_steps: 100
```

### 6.2 Hulk 框架配置
```yaml
model:
  _name: model_parallel_whisper_distil_dec
  from_pretrained: openai/whisper-small

  n_audio_ctx: 1500
  n_audio_state: 1280
  n_text_state: 1280
  n_audio_layer: 32
  n_text_layer: 4

optimization:
  max_epoch: 3
  lr: [0.00005]
  clip_norm: 12.0

lora:
  apply_lora: true
  lora_rank: 16
  lora_alpha: 128
```

---

## 7. Hulk 框架训练流程

### 7.1 数据准备
```bash
python scripts/prepare_hulk_data.py \
  --output_dir ./data/hulk_lmdb \
  --generate_synthetic \
  --num_samples 10000
```

生成文件:
- `hungarian_lmdb/` - LMDB 数据库 (500GB map_size)
- `hungarian_chunk10000.bin` - Chunk 索引文件
- `hungarian_dataset.json` - 数据集描述 JSON
- `phone_dict.json` - 音素字典 (48 phones + blank)

### 7.2 配置生成
```bash
bash scripts/train_hulk.sh
```

输出:
- `output/hulk_hungarian/whisper_train_hungarian.yaml` - Hulk 配置
- `output/hulk_hungarian/submit_hu_ED.sh` - 提交脚本

### 7.3 提交训练
```bash
# 单机多卡
cd /home/lty/hungarian_whisper
torchrun --nproc_per_node=8 \
  /home/lty/am/whisper_ED/hulk/hulk_cli/hydra_train.py \
  --config-dir ./output/hulk_hungarian \
  --config-name whisper_train_hungarian.yaml

# 集群提交 (需公司HPC环境)
bash output/hulk_hungarian/submit_hu_ED.sh
```

---

## 8. 平台支持

| 组件 | CUDA (NVIDIA) | ROCm (AMD) |
|------|---------------|-------------|
| PyTorch | ✅ torch (CUDA) | ✅ torch+rocm5.7 |
| INT8 量化 | ✅ bitsandbytes | ❌ 不支持 |
| BF16/FP16 | ✅ | ✅ |
| LoRA | ✅ | ✅ |
| Hulk 框架 | ✅ | ✅ |

---

## 9. WFST 解码准备

### 9.1 解码图结构
CTC 和 ED 模型训练完成后，需连接 WFST 进行解码：

```
L.fst (词汇表) → G.fst (语言模型) → C.fst (上下文) → T.fst (发音)
最终生成 HCLG.fst 解码图
```

### 9.2 匈牙利语 WFST 资源
- **音素集**: 48 phones (见 hungarian_normalizer.py)
- **词汇表**: 需从 Hungarian lexicon 构建
- **语言模型**: 需训练 Hungarian LM 或使用现有资源

### 9.3 解码配置
```yaml
decoder:
  name: "wfst"
  beam_size: 20
  lattice_beam: 10.0
  word_insertion_penalty: 0.0
  lexicon:
    fst: "L.fst"
  language_model:
    fst: "G.fst"
```

---

## 10. 项目结构

```
hungarian_whisper/
├── config/
│   ├── config.yaml              # CUDA 配置
│   └── config_rocm.yaml         # ROCm 配置
├── src/
│   ├── data/
│   │   ├── hungarian_normalizer.py   # 文本标准化
│   │   ├── htk_exporter.py           # HTK导出
│   │   ├── htk_dataloader.py         # HTK数据加载
│   │   └── dataset_loader.py         # HF数据集加载
│   ├── model/
│   │   └── lora_whisper.py          # LoRA模型配置
│   ├── training/
│   │   ├── trainer.py               # 训练循环
│   │   └── evaluation.py            # 评估指标
│   └── utils/
│       └── memory_monitor.py        # VRAM监控
├── scripts/
│   ├── train_hulk.sh              # Hulk 训练配置生成
│   ├── prepare_hulk_data.py       # LMDB 数据准备
│   └── train_rocm_simple.py       # ROCm 快速训练
├── data/
│   ├── hulk_lmdb/                 # LMDB 格式数据
│   │   ├── hungarian_lmdb/
│   │   ├── hungarian_chunk10000.bin
│   │   ├── hungarian_dataset.json
│   │   └── phone_dict.json
│   └── htk_output/                # HTK 格式数据
├── output/
│   ├── checkpoints/              # ROCm 训练的模型
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── training_summary.json
│   └── hulk_hungarian/           # Hulk 配置
│       ├── whisper_train_hungarian.yaml
│       └── submit_hu_ED.sh
└── Hungarian_Whisper_Tuning_Report.md
```

---

## 11. 依赖

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
hydra-core
omegaconf
lmdb
```

---

## 12. 使用说明

### 12.1 ROCm 训练 (已验证)
```bash
cd /home/lty/hungarian_whisper
python scripts/train_rocm_simple.py
```

### 12.2 Hulk 数据准备
```bash
python scripts/prepare_hulk_data.py \
  --output_dir ./data/hulk_lmdb \
  --generate_synthetic \
  --num_samples 10000
```

### 12.3 Hulk 训练提交
```bash
bash scripts/train_hulk.sh
# 然后在集群上运行
bash output/hulk_hungarian/submit_hu_ED.sh
```

---

## 13. 注意事项

1. **ROCm INT8**: bitsandbytes 不支持 ROCm，使用 BF16 替代
2. **Hulk 环境**: 需要公司 HPC 集群环境 (Python 3.8)
3. **HTK 兼容性**: 导出的HTK文件可用于HTK Toolkit进行进一步处理
4. **监控**: 训练过程中持续监控VRAM使用情况

---

## 14. 下一步

1. **Hulk 集群训练**: 在公司 HPC 环境运行 `submit_hu_ED.sh`
2. **WFST 解码**: 准备 Hungarian lexicon 和 LM，构建 HCLG 图
3. **模型评估**: 在验证集上计算 WER
4. **生产部署**: 导出为 ONNX 或 TorchScript

---

## 15. 参考资料

- [Whisper论文](https://arxiv.org/abs/2212.04356)
- [PEFT Library](https://github.com/huggingface/peft)
- [HTK Toolkit](http://htk.eng.cam.ac.uk/)
- [Common Voice数据集](https://huggingface.co/datasets/mozilla-foundation/common_voice)
- [VoxPopuli数据集](https://huggingface.co/datasets/facebook/voxpopuli)
- [FLEURS数据集](https://huggingface.co/datasets/google/fleurs)
- [Hulk Framework](file:///home/lty/am/whisper_ED/hulk)

---

*报告生成时间: 2026-04-20*