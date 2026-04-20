#!/usr/bin/env python3
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Clear proxies
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(k, None)

import torch
import torch.nn as nn

print(f"PyTorch: {torch.__version__}")
print(f"ROCm: {torch.version.hip}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Simple linear test
m = nn.Linear(100, 100).cuda()
x = torch.randn(32, 100).cuda()
y = torch.randn(32, 100).cuda()

loss = nn.functional.mse_loss(m(x), y)
loss.backward()
optimizer = torch.optim.Adam(m.parameters())
optimizer.step()
print("Simple linear test OK")

# Test with Whisper forward only (no training)
from transformers import WhisperForConditionalGeneration

print("Loading Whisper...")
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
model = model.cuda()
model.eval()

print("Testing Whisper forward (eval mode, no gradients)...")
x = torch.randn(1, 80, 3000).cuda()
with torch.no_grad():
    out = model(input_features=x)
print(f"Forward OK, logits shape: {out.logits.shape}")

# Test training mode without backward
print("Testing Whisper forward in train mode (no backward)...")
model.train()
x = torch.randn(1, 80, 3000).cuda()
labels = torch.randint(0, 51865, (1, 10)).cuda()
out = model(input_features=x, labels=labels)
print(f"Train forward OK, loss: {out.loss.item()}")

# Full training step
print("Testing full training step (with backward)...")
model.zero_grad()
out = model(input_features=x, labels=labels)
out.loss.backward()
print("Backward OK")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.step()
print("Optimizer step OK")

print("\nAll tests passed!")