"""
inference_ai.py
===============
Extended inference pipeline for ForensicsSAM with AI-forgery detection.

Three-way classification:
  AUTHENTIC          — no forgery detected
  TRADITIONAL FORGED — copy-move / splicing / traditional inpainting
  AI FORGED          — diffusion model manipulation (Stable Diffusion,
                       SDXL, Adobe Firefly, FLUX, DALL-E, etc.)

The script outputs:
  • Console: image path, predicted class, confidence
  • Visual : side-by-side overlay saved as <input>_forensics.png

Usage — single image:
    python inference_ai.py --image /path/to/image.jpg

Usage — batch from dataset list:
    python inference_ai.py --batch
"""

import os
import argparse
import random
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# ── Project imports ────────────────────────────────────────────────────────
from segment_anything import sam_model_registry
from forensics_sam import ForensicsSAM
from adversary_detector import AdversaryDetector
from ai_forgery_detector import AIForgeryClassifier
from mini_dataloader import TGIFDataloader
from utils import AverageMeter, cal_f1, cal_auc, cal_permute_f1, cal_image_level_acc

# ===========================================================================
# ░░░  CONFIGURATION  ░░░
# ===========================================================================

TGIF_ROOT = "./data/tgif"      # ← set before running

SAM_TYPE   = "vit_h"
CHECKPOINT = {
    "vit_b": "./weight/sam_vit_b_01ec64.pth",
    "vit_l": "./weight/sam_vit_l_0b3195.pth",
    "vit_h": "./weight/sam_vit_h_4b8939.pth",
}
IMAGE_SIZE    = 1024
NORMALIZE_TYPE = 2
LORA_RANK     = 8
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHT_DIR = "./weight"
ADV_WEIGHT_PATH = os.path.join(WEIGHT_DIR, "adversary_detector.pth")
AI_DET_WEIGHT   = os.path.join(WEIGHT_DIR, "ai_forgery_detector.pth")
AI_LORA_WEIGHT  = os.path.join(WEIGHT_DIR, "ai_forgery_experts.pth")

# Output folder for visualisations
VIZ_DIR = "./output/forensics"

# Confidence thresholds
ADV_THRESHOLD    = 0.5   # above → adversarial perturbation
AI_THRESHOLD     = 0.5   # above → AI-generated forgery
FORGED_THRESHOLD = 0.5   # above → any forgery present

# Label strings
CLASS_NAMES = {
    "authentic":    "AUTHENTIC",
    "traditional":  "TRADITIONAL FORGERY",
    "ai_forged":    "AI-GENERATED FORGERY",
    "adversarial":  "ADVERSARIAL + FORGERY",
}

# Overlay colours (BGR)
MASK_COLOUR_TRADITIONAL = (0, 0, 220)   # red
MASK_COLOUR_AI          = (0, 165, 255) # orange
MASK_COLOUR_ADV         = (220, 0, 220) # purple
MASK_ALPHA              = 0.45

SEED = 42

# ===========================================================================
# ░░░  HELPERS  ░░░
# ===========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image_tensor(image_path: str, image_size: int = 1024) -> torch.Tensor:
    """
    Read an image file, resize, convert BGR→RGB, normalise (ImageNet),
    and return a (1, 3, H, W) tensor.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (image_size, image_size), cv2.INTER_AREA)
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB [0,1]
    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def overlay_mask(image_bgr: np.ndarray,
                 mask: np.ndarray,
                 colour: tuple,
                 alpha: float = 0.45) -> np.ndarray:
    """
    Blend a binary mask onto an image using the given colour.

    Parameters
    ----------
    image_bgr : (H, W, 3) uint8
    mask      : (H, W) float/uint8, values in [0,1] or {0,255}
    colour    : BGR tuple
    """
    overlay = image_bgr.copy()
    mask_bool = (mask > 0.5).astype(bool)
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(colour)
    ).clip(0, 255).astype(np.uint8)
    return overlay


def build_result_panel(orig: np.ndarray,
                       mask_full: np.ndarray,
                       label: str,
                       score: float,
                       mask_colour: tuple) -> np.ndarray:
    """Create a side-by-side result panel: original | masked overlay."""
    h, w = orig.shape[:2]

    # Resize mask to match original
    mask_resized = cv2.resize(
        (mask_full * 255).astype(np.uint8), (w, h), cv2.INTER_NEAREST
    )
    overlaid = overlay_mask(orig, mask_resized / 255.0, mask_colour, MASK_ALPHA)

    # Add label text
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_size = max(0.5, h / 800)
    thickness = max(1, int(h / 400))
    cv2.putText(overlaid, label,         (10, 30),  font, font_size, (255,255,255), thickness)
    cv2.putText(overlaid, f"{score:.1%}",(10, 60),  font, font_size, (200,255,200), thickness)

    panel = np.hstack([orig, overlaid])
    return panel


# ===========================================================================
# ░░░  INFERENCE FUNCTION  ░░░
# ===========================================================================

@torch.no_grad()
def predict_single(
    image_path: str,
    forensics_sam: ForensicsSAM,
    adv_detector:  AdversaryDetector,
    ai_detector:   AIForgeryClassifier,
    save_viz: bool = True,
) -> dict:
    """
    Run full three-stage inference on a single image.

    Returns a dict with keys:
        label        : str  ('authentic' | 'traditional' | 'ai_forged' | 'adversarial')
        adv_score    : float  (probability of adversarial perturbation)
        ai_score     : float  (probability of AI-generated forgery)
        forged_score : float  (probability of being forged, any type)
        mask         : np.ndarray  (H//4, W//4) binary mask
    """
    image_tensor = load_image_tensor(image_path, IMAGE_SIZE).to(DEVICE)

    # Stage 1: Adversarial detection
    adv_logits, _ = adv_detector(image_tensor)
    adv_probs     = torch.softmax(adv_logits, dim=1)
    adv_pred      = int(torch.argmax(adv_probs, dim=1).item())
    adv_score     = adv_probs[0, 1].item()

    # Stage 2: AI-forgery detection
    ai_logits, _ = ai_detector(image_tensor)
    ai_probs     = torch.softmax(ai_logits, dim=1)
    ai_pred      = int(torch.argmax(ai_probs, dim=1).item())
    ai_score     = ai_probs[0, 1].item()

    # Stage 3: ForensicsSAM segmentation (activate appropriate LoRA)
    activate_adv = (adv_score > ADV_THRESHOLD)
    activate_ai  = (ai_score  > AI_THRESHOLD) and not activate_adv

    mask_pred, cls_pred = forensics_sam(
        image_tensor,
        activate_adv=activate_adv,
        activate_ai=activate_ai,
    )

    mask_logits  = torch.sigmoid(mask_pred)        # (1,1,H',W')
    cls_logits   = torch.sigmoid(cls_pred)         # (1,1)
    forged_score = cls_logits.squeeze().item()

    mask_binary  = (mask_logits > FORGED_THRESHOLD).squeeze().cpu().numpy().astype(np.float32)

    # Determine top-level label
    if forged_score < FORGED_THRESHOLD:
        label = "authentic"
        mask_colour = (128, 128, 128)
    elif activate_adv:
        label = "adversarial"
        mask_colour = MASK_COLOUR_ADV
    elif activate_ai:
        label = "ai_forged"
        mask_colour = MASK_COLOUR_AI
    else:
        label = "traditional"
        mask_colour = MASK_COLOUR_TRADITIONAL

    # Visualisation
    if save_viz:
        os.makedirs(VIZ_DIR, exist_ok=True)
        orig = cv2.imread(image_path)
        if orig is not None:
            panel = build_result_panel(
                orig, mask_binary,
                label=CLASS_NAMES[label],
                score=max(ai_score, adv_score, forged_score),
                mask_colour=mask_colour,
            )
            stem = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(VIZ_DIR, f"{stem}_forensics.png")
            cv2.imwrite(out_path, panel)

    return {
        "label":        label,
        "adv_score":    adv_score,
        "ai_score":     ai_score,
        "forged_score": forged_score,
        "mask":         mask_binary,
    }


# ===========================================================================
# ░░░  BATCH EVALUATION (TGIF dataset)  ░░░
# ===========================================================================

@torch.no_grad()
def evaluate_batch(
    forensics_sam: ForensicsSAM,
    adv_detector:  AdversaryDetector,
    ai_detector:   AIForgeryClassifier,
):
    """Run batch evaluation on TGIF validation set and report metrics."""

    val_dataset = TGIFDataloader(
        dataset_list=[
            [TGIF_ROOT, "tgif_val.txt",    1.0, 1, 1, 0, 1],
            [TGIF_ROOT, "tgif_au_val.txt", 1.0, 1, 0, 0, 0],
        ],
        input_size=IMAGE_SIZE,
        normalize_type=NORMALIZE_TYPE,
        augment_prob=0.0,
        enable_aug_types=[],
        mode="val",
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    auc_m    = AverageMeter()
    f1_m     = AverageMeter()
    p_f1_m   = AverageMeter()
    ai_acc_m = AverageMeter()
    f_acc_m  = AverageMeter()

    t = tqdm(val_loader, desc="[Eval] TGIF val set")
    for images, gt_masks, forged_labels, adv_labels, ai_labels in t:
        images        = images.to(DEVICE)
        gt_masks      = gt_masks.to(DEVICE)

        # AI detector
        ai_logits, _ = ai_detector(images)
        ai_preds      = torch.argmax(ai_logits, dim=1)
        ai_probs      = torch.softmax(ai_logits, dim=1)[:, 1]

        # Adv detector
        adv_logits, _ = adv_detector(images)
        adv_preds     = torch.argmax(adv_logits, dim=1)

        # ForensicsSAM
        activate_ai  = bool(ai_preds.any().item())
        activate_adv = bool(adv_preds.any().item()) and not activate_ai
        mask_pred, cls_pred = forensics_sam(images, activate_adv, activate_ai)

        mask_logits = torch.sigmoid(mask_pred)
        mask_binary = (mask_logits > 0.5).float()
        cls_logits  = torch.sigmoid(cls_pred)

        # Metrics
        if forged_labels.squeeze() == 1:
            auc_m.update(cal_auc(mask_logits.cpu().numpy(), gt_masks.cpu().numpy()))
            f1_m.update(cal_f1(mask_binary.cpu().numpy(), gt_masks.cpu().numpy()))
            p_f1_m.update(cal_permute_f1(mask_binary.cpu().numpy(), gt_masks.cpu().numpy()))

        f_acc_m.update(
            cal_image_level_acc(cls_logits.cpu().numpy(), forged_labels.cpu().numpy(), 0.5)
        )
        ai_acc_m.update(
            cal_image_level_acc(ai_probs.cpu().numpy(), ai_labels.cpu().numpy(), 0.5)
        )

        t.set_description(
            f"AUC:{auc_m.avg:.3f} F1:{f1_m.avg:.3f} P-F1:{p_f1_m.avg:.3f} "
            f"F_ACC:{f_acc_m.avg:.3f} AI_ACC:{ai_acc_m.avg:.3f}"
        )

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  TGIF Validation Results")
    print(f"  Mask AUC    : {auc_m.avg:.4f}")
    print(f"  Mask F1     : {f1_m.avg:.4f}")
    print(f"  Mask P-F1   : {p_f1_m.avg:.4f}")
    print(f"  Forgery ACC : {f_acc_m.avg:.4f}")
    print(f"  AI-Det ACC  : {ai_acc_m.avg:.4f}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


# ===========================================================================
# ░░░  ENTRY POINT  ░░░
# ===========================================================================

def build_models():
    set_seed(SEED)
    print(f"[inference_ai] Device: {DEVICE}")

    # SAM + ForensicsSAM
    sam, _ = sam_model_registry[SAM_TYPE](
        image_size=IMAGE_SIZE,
        checkpoint=CHECKPOINT[SAM_TYPE],
    )
    forensics_sam = ForensicsSAM(
        sam,
        r=LORA_RANK,
        load_ai_weights=os.path.exists(AI_LORA_WEIGHT),
    ).to(DEVICE).eval()

    # Adversarial detector
    adv_detector = AdversaryDetector().to(DEVICE).eval()
    if os.path.exists(ADV_WEIGHT_PATH):
        adv_detector.load_detector(ADV_WEIGHT_PATH)
    else:
        print(f"[WARN] Adversarial detector weights not found at {ADV_WEIGHT_PATH}")

    # AI forgery detector
    ai_detector = AIForgeryClassifier(
        backbone="resnet18", pretrained=True
    ).to(DEVICE).eval()
    if os.path.exists(AI_DET_WEIGHT):
        ai_detector.load_detector(AI_DET_WEIGHT)
    else:
        print(f"[WARN] AI forgery detector weights not found at {AI_DET_WEIGHT}. "
              f"Run train_ai_detector.py first.")

    return forensics_sam, adv_detector, ai_detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForensicsSAM inference with AI forgery detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single input image.")
    group.add_argument("--batch",  action="store_true",
                       help="Run batch evaluation on TGIF val set.")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip saving visualisation overlays.")
    args = parser.parse_args()

    forensics_sam, adv_detector, ai_detector = build_models()

    if args.batch:
        evaluate_batch(forensics_sam, adv_detector, ai_detector)
    else:
        result = predict_single(
            image_path=args.image,
            forensics_sam=forensics_sam,
            adv_detector=adv_detector,
            ai_detector=ai_detector,
            save_viz=not args.no_viz,
        )
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Image        : {args.image}")
        print(f"  Verdict      : {CLASS_NAMES[result['label']]}")
        print(f"  Forgery conf : {result['forged_score']:.1%}")
        print(f"  AI-forge conf: {result['ai_score']:.1%}")
        print(f"  Adv conf     : {result['adv_score']:.1%}")
        mask_coverage = result["mask"].mean() * 100
        print(f"  Mask coverage: {mask_coverage:.1f}% of image")
        if not args.no_viz:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            print(f"  Visualisation: {VIZ_DIR}/{stem}_forensics.png")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
