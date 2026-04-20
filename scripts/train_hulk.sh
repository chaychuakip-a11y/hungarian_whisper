#!/bin/bash
# =================================================================
# Hungarian Whisper Training using Hulk Framework
# Based on /home/lty/am/whisper_ED
# =================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HULK_DIR="/home/lty/am/whisper_ED/hulk"

echo "========================================"
echo "Hungarian Whisper Training (Hulk Framework)"
echo "========================================"

# Data directory (LMDB format)
DATA_DIR="$PROJECT_DIR/data/hulk_lmdb"
LMDB_PATH="$DATA_DIR/hungarian_lmdb"
CHUNK_PATH="$DATA_DIR/hungarian_chunk10000.bin"
TRAIN_JSON="$DATA_DIR/hungarian_dataset.json"
VALID_JSON="$DATA_DIR/hungarian_dataset.json"  # Use same file for validation

# Model paths
PRETRAINED_PATH="openai/whisper-small"

# Output directory
OUTPUT_DIR="$PROJECT_DIR/output/hulk_hungarian"
mkdir -p "$OUTPUT_DIR"

# Create configuration file for Hungarian
cat > "$OUTPUT_DIR/whisper_train_hungarian.yaml" << EOF
# @package _group_
hydra:
  run:
    dir: $OUTPUT_DIR/outputs
  job:
    name: hungarian_whisper

common:
  fp16: true
  memory_efficient_fp16: false
  log_format: json
  log_interval: 10
  tensorboard_logdir: $OUTPUT_DIR/tb
  fp16_no_flatten_grads: false
  cudnn_benchmark: false
  cudnn_deterministic: false
  seed: 42
  user_dir: $PROJECT_DIR

checkpoint:
  save_interval: 1
  keep_interval_updates: 10
  no_epoch_checkpoints: false

task:
  _name: multilingual_asr_task
  nmod: 1
  min_sample_size: 60
  max_sample_size: 1500
  feature_type: "fb80"
  use_poi: 0
  chunk_seq_path: "$CHUNK_PATH"

dataset:
  do_debug: false
  language_add_space: hu

  skip_invalid_size_inputs_valid_test: true
  num_workers: 4
  batch_size: 512
  max_tokens: 4096
  validate_interval: 1
  validate_interval_updates: 5000
  disable_validation: false

  train_subset: "$TRAIN_JSON"
  valid_subset: "$VALID_JSON"
  shuffle: true
  num_parts: 1
  num_blocks: 50
  block_size: 1000
  seed: 1000

model_parallel:
  micro_batch_size: 128
  num_micro_batch: 4
  pipeline_model_parallel_size: 1
  tensor_model_parallel_size: 1
  use_cpu_initialization: true

distributed_training:
  ddp_backend: pytorch_ddp

criterion:
  _name: loss

optimization:
  max_epoch: 3
  clip_norm: 12.0
  lr: [0.00005]
  update_freq: [1]
  skip_remainder_batch: true

optimizer:
  _name: adam
  adam_betas: (0.9, 0.98)
  adam_eps: 1e-6
  weight_decay: 0.1

lr_scheduler:
  _name: polynomial_decay
  warmup_updates: 500
  total_num_update: 100000

model:
  _name: model_parallel_whisper_distil_dec
  n_vocab: 51865
  mlu: false
  reset_optimizer: true
  reset_dataloader: true

  from_pretrained: "$PRETRAINED_PATH"

  n_audio_ctx: 1500
  n_audio_state: 1280
  n_text_state: 1280
  n_audio_head: 20
  n_text_head: 20
  n_audio_layer: 32
  n_text_layer: 4
  use_flash_attention: false
  parallel_output: false
  scaled_masked_softmax_fusion: false
  attention_dropout_p: 0.0
  hidden_dropout_p: 0.0

lora:
  apply_lora: true
  lora_rank: 16
  lora_alpha: 128
  lora_dropout: 0.05
  adapt_q: true
  adapt_k: true
  adapt_v: true
  adapt_o: true
  adapt_fc1: false
  adapt_fc2: false
  merge_weights: false
EOF

echo "========================================="
echo "Hulk Configuration Generated"
echo "========================================="
echo "Config: $OUTPUT_DIR/whisper_train_hungarian.yaml"
echo ""
echo "Data paths:"
echo "  LMDB: $LMDB_PATH"
echo "  Chunk: $CHUNK_PATH"
echo "  Train JSON: $TRAIN_JSON"
echo "  Valid JSON: $VALID_JSON"
echo ""
echo "Model: $PRETRAINED_PATH"
echo ""
echo "========================================="
echo "USAGE INSTRUCTIONS"
echo "========================================="
echo ""
echo "The YAML config has been generated with real paths."
echo ""
echo "To run training on the company cluster:"
echo ""
echo "  1. Copy the config to your job submission"
echo "  2. Modify DIST_ARGS for your cluster setup"
echo "  3. Submit using: bash submit_hu_ED.sh"
echo ""
echo "Or run manually on a GPU server:"
echo "  cd $PROJECT_DIR"
echo "  torchrun --nproc_per_node=8 $HULK_DIR/hulk_cli/hydra_train.py \\"
echo "    --config-dir $OUTPUT_DIR \\"
echo "    --config-name whisper_train_hungarian.yaml"
echo ""
echo "========================================="

# Create a submit script for easy deployment
cat > "$OUTPUT_DIR/submit_hu_ED.sh" << 'SUBMIT_EOF'
#!/bin/bash
# Submit Hungarian Whisper training job
# Usage: bash submit_hu_ED.sh

PROJECT_DIR="/home/lty/hungarian_whisper"
HULK_DIR="/home/lty/am/whisper_ED/hulk"
OUTPUT_DIR="$PROJECT_DIR/output/hulk_hungarian"

# Cluster configuration
# Adjust these for your cluster environment
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export WORLD_SIZE=1
export RANK=0

# Multi-GPU settings
export NPROC_PER_NODE=8
export NCCL_IB_DISABLE=0

torchrun \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --nproc_per_node=${NPROC_PER_NODE} \
    $HULK_DIR/hulk_cli/hydra_train.py \
    --config-dir $OUTPUT_DIR \
    --config-name whisper_train_hungarian.yaml \
    distributed_training.ddp_backend=pytorch_ddp
SUBMIT_EOF

chmod +x "$OUTPUT_DIR/submit_hu_ED.sh"
echo "Submit script: $OUTPUT_DIR/submit_hu_ED.sh"