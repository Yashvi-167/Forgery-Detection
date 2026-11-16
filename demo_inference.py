"""
demo_inference.py
=================
Run ForensicsSAM + AI Forgery Classifier on a SINGLE IMAGE.
No dataset merging, no GPU, no text lists required.

Usage:
    python demo_inference.py --image <path_to_image>

Example:
    python demo_inference.py --image "D:/ForensicSAME/train/fake/0001.jpg"
    python demo_inference.py --image "D:/ForensicSAME/train/real/0001.jpg"
"""

import os
import sys
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU

import torch
import torch.nn.functional as F
import cv2

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHT_DIR        = "./weight"
SAM_CHECKPOINT    = os.path.join(WEIGHT_DIR, "sam_vit_h_4b8939.pth")
ADV_DET_WEIGHT    = os.path.join(WEIGHT_DIR, "adversary_detector.pth")
FORGERY_EXP_WEIGHT= os.path.join(WEIGHT_DIR, "forgery_experts.pth")
AI_CLS_WEIGHT     = os.path.join(WEIGHT_DIR, "ai_forgery_detector.pth")

IMAGE_SIZE    = 1024      # SAM native resolution
NORMALIZE_TYPE= 2         # SAM-style normalisation

# ── SAM normalisation constants ───────────────────────────────────────────────
PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
PIXEL_STD  = torch.tensor([58.395,  57.12,  57.375]).view(3, 1, 1)


def preprocess_image(img_bgr: np.ndarray, size: int = IMAGE_SIZE):
    """Resize + pad + normalise to (1, 3, size, size) float32 tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = img_rgb.shape[:2]
    # Resize longest side to `size`
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_rgb = cv2.resize(img_rgb, (nw, nh))
    # Pad to square
    padded = np.zeros((size, size, 3), dtype=np.float32)
    padded[:nh, :nw] = img_rgb
    # HWC → CHW → (1, 3, H, W)
    t = torch.from_numpy(padded).permute(2, 0, 1)   # (3, H, W)
    t = (t - PIXEL_MEAN) / PIXEL_STD
    return t.unsqueeze(0)   # (1, 3, H, W)


def run_ai_classifier(image_path: str):
    """Run our custom AI-forgery binary classifier only (fast, CPU)."""
    print("\n[AI Forgery Classifier]")
    try:
        from ai_forgery_detector import AIForgeryClassifier
        model = AIForgeryClassifier()
        ckpt = torch.load(AI_CLS_WEIGHT, map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()
    except Exception as e:
        print(f"  Could not load AI classifier: {e}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"  Cannot read image: {image_path}")
        return

    # Simple 224×224 resize for the classifier (no SAM padding needed)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_resized = cv2.resize(img_rgb, (224, 224))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized - mean) / std
    t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        out  = model(t)
        # model may return (logit,) tuple or plain tensor
        logit = out[0] if isinstance(out, (tuple, list)) else out
        if logit.dim() > 1 and logit.shape[-1] > 1:
            # multi-class: take class-1 probability
            prob = torch.softmax(logit, dim=-1)[0, 1].item()
        else:
            prob = torch.sigmoid(logit.view(-1)[0]).item()
        label = "AI-GENERATED / FAKE" if prob > 0.5 else "REAL / AUTHENTIC"

    print(f"  Image : {os.path.basename(image_path)}")
    print(f"  Result: {label}")
    print(f"  Confidence: {prob*100:.1f}% AI-fake" if prob > 0.5
          else f"  Confidence: {(1-prob)*100:.1f}% real")


def run_forensics_sam(image_path: str, save_mask: bool = True):
    """Run full ForensicsSAM: adversary detection + forgery localization."""
    print("\n[ForensicsSAM — Forgery Localization]")

    # ── Check weights ─────────────────────────────────────────────────────────
    for name, path in [("SAM backbone", SAM_CHECKPOINT),
                        ("Adversary detector", ADV_DET_WEIGHT),
                        ("Forgery experts", FORGERY_EXP_WEIGHT)]:
        if not os.path.exists(path):
            print(f"  Missing {name}: {path}")
            return
        print(f"  OK  {name} ({os.path.getsize(path)//1024//1024} MB)")

    # ── Load image ────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  Cannot read: {image_path}")
        return
    h_orig, w_orig = img_bgr.shape[:2]
    img_tensor = preprocess_image(img_bgr)   # (1, 3, 1024, 1024)
    print(f"  Input : {os.path.basename(image_path)}  [{w_orig}×{h_orig}]")

    # ── Load models ───────────────────────────────────────────────────────────
    print("  Loading SAM backbone (may take ~30 s on CPU)...")
    from segment_anything import sam_model_registry
    from forensics_sam import ForensicsSAM
    from adversary_detector import AdversaryDetector

    sam, _ = sam_model_registry["vit_h"](
        image_size=IMAGE_SIZE, checkpoint=SAM_CHECKPOINT)
    sam = sam.cpu().eval()

    model = ForensicsSAM(sam, r=8).cpu().eval()

    # Load forgery experts
    fe_ckpt = torch.load(FORGERY_EXP_WEIGHT, map_location="cpu")
    missing, unexpected = model.load_state_dict(fe_ckpt, strict=False)
    print(f"  Forgery experts loaded  (missing={len(missing)}, unexpected={len(unexpected)})")

    adv_det = AdversaryDetector().cpu().eval()
    adv_det.load_detector(ADV_DET_WEIGHT)
    print("  Adversary detector loaded")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("  Running inference (CPU — may take 1–2 min)...")
    with torch.no_grad():
        adv_logits, _ = adv_det(img_tensor)
        adv_pred = torch.argmax(adv_logits, dim=1)
        is_adversarial = adv_pred.item() == 1

        mask_logit, cls_logit = model(img_tensor, adv_pred)

        mask_prob = torch.sigmoid(mask_logit)
        mask_bin  = (mask_prob > 0.5).float()
        cls_prob  = torch.sigmoid(cls_logit).item()
        is_forged = cls_prob > 0.5

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n  === RESULTS ===")
    print(f"  Image-level : {'FORGED' if is_forged else 'AUTHENTIC'}"
          f"  (confidence {cls_prob*100:.1f}%)")
    print(f"  Adversarial : {'YES (attacked image)' if is_adversarial else 'No'}")

    # Compute forged area %
    mask_np = mask_bin.squeeze().cpu().numpy()
    # Resize mask back to original image size
    mask_up = cv2.resize(mask_np, (w_orig, h_orig))
    forged_pct = mask_up.mean() * 100
    print(f"  Forged area : {forged_pct:.1f}% of image")

    # ── Save visualisation ────────────────────────────────────────────────────
    if save_mask:
        out_dir = "demo_output"
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]

        # Save binary mask
        mask_save = (mask_up * 255).astype(np.uint8)
        mask_path = os.path.join(out_dir, f"{base}_mask.png")
        cv2.imwrite(mask_path, mask_save)

        # Overlay: red = forged regions
        overlay = img_bgr.copy()
        overlay[mask_up > 0.5] = overlay[mask_up > 0.5] * 0.4 + \
                                  np.array([0, 0, 200]) * 0.6
        overlay_path = os.path.join(out_dir, f"{base}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay.astype(np.uint8))

        print(f"\n  Saved mask   : {mask_path}")
        print(f"  Saved overlay: {overlay_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ForensicsSAM single-image demo")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--mode",   default="both",
                        choices=["classify", "localize", "both"],
                        help="classify=AI classifier only, localize=SAM only, both=all")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output masks")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print("  ForensicsSAM Demo — Single Image Inference")
    print("=" * 60)
    print(f"  Image : {args.image}")
    print(f"  Mode  : {args.mode}")

    if args.mode in ("classify", "both"):
        run_ai_classifier(args.image)

    if args.mode in ("localize", "both"):
        run_forensics_sam(args.image, save_mask=not args.no_save)

    print("\nDone.")


if __name__ == "__main__":
    main()
