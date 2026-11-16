"""Quick validation of all new delta-addition modules."""
import sys, os, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

print("=" * 60)
print("  ForensicsSAM AI-Forgery Delta Addition — Validation")
print("=" * 60)

# 1. AIForgeryClassifier
print("\n[1] AIForgeryClassifier ...")
from ai_forgery_detector import AIForgeryClassifier
m = AIForgeryClassifier(backbone="resnet18", pretrained=False)
x = torch.randn(1, 3, 224, 224)
logits, feats = m(x)
assert logits.shape == (1, 2), f"Expected (1,2), got {logits.shape}"
assert feats.shape == (1, 640), f"Expected (1,640), got {feats.shape}"
print("    logits:", logits.shape, " feats:", feats.shape, " PASS")

# 2. TGIFDataloader
print("\n[2] TGIFDataloader ...")
from mini_dataloader import TGIFDataloader, BasicDataloader
print("    Import OK  PASS")

# 3. AI_LoRA modules
print("\n[3] AI_LoRA_QKV / AI_LoRA_FFN ...")
from forensics_sam import AI_LoRA_QKV, AI_LoRA_FFN
dim = 64
r = 8
qkv = AI_LoRA_QKV(nn.Linear(dim, dim * 3), dim, r)
ffn = AI_LoRA_FFN(nn.Linear(dim, dim * 4), dim, dim * 4, r)

# Test with activate=False (should just pass through)
x2 = torch.randn(1, 4, 4, dim)
out1 = qkv(x2)
assert out1.shape == (1, 4, 4, dim * 3)

# Test with activate=True (should add LoRA delta)
qkv.set_activate(True)
ffn.set_activate(True)
out2 = qkv(x2)
assert out2.shape == (1, 4, 4, dim * 3)

x3 = torch.randn(1, dim)
out3 = ffn(x3)
assert out3.shape == (1, dim * 4)
print("    QKV:", out2.shape, " FFN:", out3.shape, " PASS")

# 4. ForensicsSAM class has new methods
print("\n[4] ForensicsSAM new methods ...")
from forensics_sam import ForensicsSAM
for method in ["activate_ai", "save_ai_parameters", "load_ai_parameters", "set_ai_lora_trainable"]:
    assert hasattr(ForensicsSAM, method), f"Missing method: {method}"
    print(f"    {method}() exists")
print("    PASS")

# 5. File existence
print("\n[5] New files ...")
files = [
    "ai_forgery_detector/__init__.py",
    "ai_forgery_detector/classifier.py",
    "mini_dataloader/tgif_dataloader.py",
    "train_ai_detector.py",
    "train_ai_lora.py",
    "inference_ai.py",
    "scripts/prepare_tgif_index.py",
]
for f in files:
    status = "EXISTS" if os.path.exists(f) else "MISSING"
    print(f"    {f}: {status}")
    assert os.path.exists(f), f"File missing: {f}"
print("    PASS")

# 6. Verify existing inference.py still importable
print("\n[6] Original inference.py compatibility ...")
# Just check the original modules still work
from forensics_sam import ForgeryDetector, LoRA_QKV, LoRA_FFN, Adv_LoRA_QKV, Adv_LoRA_FFN
from adversary_detector import AdversaryDetector
print("    All original classes importable  PASS")

print("\n" + "=" * 60)
print("  ALL CHECKS PASSED")
print("=" * 60)
