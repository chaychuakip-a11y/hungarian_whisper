# 匈牙利语Whisper模型微调实验报告

## 1. 项目概述

### 1.1 目标
在本地环境中对OpenAI Whisper模型进行匈牙利语微调，使用HTK（Hidden Markov Model Toolkit）标注格式，处理24GB VRAM约束下的训练任务。

### 1.2 关键约束
- **模型**: whisper-large-v2 (1.5B参数)
- **量化**: INT8量化 + LoRA微调
- **VRAM限制**: 24GB
- **数据集**: Common Voice + VoxPopuli + FLEERS（匈牙利语）
- **输出格式**: 训练后的模型 + HTK格式文件

---

## 2. 数据清洗规则

### 2.1 匈牙利语字符集
保留的有效匈牙利语字符：
- 小写: `aáäbcdeééfghiíjkloóöőpqrstuúüűvwxyz`
- 大写: `AÁÄBCDEÉÉFGHIÍJKLOÓÖŐPQRSTUÚÜŰVWXYZ`
- 特殊字符: `á, é, í, ó, ö, ő, ú, ü, ű`

### 2.2 文本标准化流程
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

### 2.3 数据过滤规则
| 条件 | 阈值 |
|------|------|
| 最小音频时长 | 1.0秒 |
| 最大音频时长 | 25.0秒 |
| 最小文本长度 | 2个单词 |
| 禁用纯数字文本 | 是 |

---

## 3. HTK格式转换逻辑

### 3.1 HTK文件格式

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

### 3.2 转换流程
1. 从HuggingFace加载原始数据集
2. 应用匈牙利语文本标准化
3. 按音频时长过滤 (1-25秒)
4. 验证文本有效性
5. 生成wav.scp文件
6. 生成labels.mlf文件
7. 保存音频文件到统一目录

---

## 4. LoRA配置详情

### 4.1 模型配置
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

### 4.2 可训练参数分析
| 参数类别 | 数量 | 占比 |
|----------|------|------|
| 全部参数 | 1.5B | 100% |
| LoRA参数 | ~4M | 0.27% |

**LoRA模块**: q_proj (查询投影), v_proj (值投影)

### 4.3 VRAM估算
| 组件 | 估算VRAM |
|------|----------|
| 模型权重 (INT8) | ~2GB |
| 激活值 (BF16) | ~8GB |
| 梯度 (BF16) | ~2GB |
| 优化器状态 (FP32) | ~8GB |
| **总计** | ~20GB |

---

## 5. 训练配置

### 5.1 训练参数
```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  effective_batch_size: 16
  learning_rate: 1e-3
  num_train_epochs: 3
  warmup_steps: 100
  bf16: true
  fp16: false
```

### 5.2 训练策略
- **精度**: BF16 (Ampere架构+)
- **梯度累积**: 4步 (有效批大小16)
- **学习率调度**: 线性预热 + 余弦衰减
- **早停**: 基于WER监控

---

## 6. 训练日志摘要

### 6.1 数据集统计
| 数据集 | 原始样本 | 过滤后 | 保留率 |
|--------|----------|--------|--------|
| Common Voice (hu) | ~XX,XXX | ~XX,XXX | XX% |
| VoxPopuli (hu) | ~XX,XXX | ~XX,XXX | XX% |
| FLEERS (hu_hu) | ~XX,XXX | ~XX,XXX | XX% |

### 6.2 训练过程
```
Step 100: Loss = X.XXX, VRAM = XX.XGB
Step 200: Loss = X.XXX, VRAM = XX.XGB
Step 500: Eval WER = X.XXX
Step 1000: Loss = X.XXX, VRAM = XX.XGB
...
```

### 6.3 VRAM监控
```
| 阶段 | 已分配 | 已缓存 | 峰值 |
|------|--------|--------|------|
| 训练前 | XX.XXGB | XX.XXGB | - |
| 训练中 | XX.XXGB | XX.XXGB | XX.XXGB |
| 训练后 | XX.XXGB | XX.XXGB | XX.XXGB |
```

---

## 7. 最终评估指标

### 7.1 WER (Word Error Rate)
| 指标 | 基础模型 | 微调后 |
|------|----------|--------|
| WER | X.XX% | X.XX% |
| 相对提升 | - | XX% |

### 7.2 其他指标
| 指标 | 值 |
|------|---|
| CER (Character Error Rate) | X.XX% |
| 实时因子 (RTF) | X.XXX |

### 7.3 示例输出
```
参考: "köszönöm szépen a figyelmet"
预测: "köszönöm szépen a figyelmet"
WER: 0.00%

参考: "ez egy hosszú magyar mondat"
预测: "ez egy hosszú magyar mondás"
WER: 16.67%
```

---

## 8. 项目结构

```
hungarian_whisper/
├── requirements.txt              # Python依赖
├── config/
│   └── config.yaml              # 配置文件
├── src/
│   ├── data/
│   │   ├── hungarian_normalizer.py   # 文本标准化
│   │   ├── htk_exporter.py           # HTK导出
│   │   ├── htk_dataloader.py         # HTK数据加载
│   │   ├── dataset_loader.py         # HF数据集加载
│   │   └── collator.py               # 数据整理器
│   ├── model/
│   │   └── lora_whisper.py          # LoRA模型配置
│   ├── training/
│   │   ├── trainer.py               # 训练循环
│   │   └── evaluation.py            # 评估指标
│   └── utils/
│       └── memory_monitor.py        # VRAM监控
├── scripts/
│   ├── 01_install_deps.sh          # 安装依赖
│   ├── 02_prepare_data.sh          # 数据准备
│   ├── 03_train.sh                 # 模型训练
│   └── 04_evaluate.sh              # 模型评估
├── data/
│   └── htk_output/                 # HTK格式输出
│       ├── wav.scp
│       └── labels.mlf
├── output/
│   ├── checkpoints/                # 训练检查点
│   └── reports/                     # 评估报告
└── Hungarian_Whisper_Tuning_Report.md
```

---

## 9. 使用说明

### 9.1 环境安装
```bash
cd hungarian_whisper
bash scripts/01_install_deps.sh
```

### 9.2 数据准备
```bash
bash scripts/02_prepare_data.sh
```

### 9.3 模型训练
```bash
bash scripts/03_train.sh
```

### 9.4 模型评估
```bash
bash scripts/04_evaluate.sh
```

---

## 10. 注意事项

1. **VRAM限制**: 24GB VRAM下，INT8量化 + LoRA是必要配置
2. **HTK兼容性**: 导出的HTK文件可用于HTK Toolkit进行进一步处理
3. **数据质量**: 严格的文本标准化对Tokenizer训练至关重要
4. **监控**: 训练过程中持续监控VRAM使用情况

---

## 11. 参考资料

- [Whisper论文](https://arxiv.org/abs/2212.04356)
- [PEFT Library](https://github.com/huggingface/peft)
- [HTK Toolkit](http://htk.eng.cam.ac.uk/)
- [Common Voice数据集](https://huggingface.co/datasets/mozilla-foundation/common_voice)
- [VoxPopuli数据集](https://huggingface.co/datasets/facebook/voxpopuli)
- [FLEERS数据集](https://huggingface.co/datasets/google/fleurs)

---

*报告生成时间: 2026-04-20*
