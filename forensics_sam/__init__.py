"""
ForensicsSAM — Extended with AI-Forgery Detection Delta Addition
================================================================

Original architecture (unchanged):
  • LoRA_QKV / LoRA_FFN           — shared forgery experts (frozen)
  • Adv_LoRA_QKV / Adv_LoRA_FFN  — adversarial robustness experts
  • ForgeryDetector               — image-level binary classifier

Delta addition (new):
  • AI_LoRA_QKV / AI_LoRA_FFN    — AI-generated forgery experts
  • activate_ai() method          — toggle AI expert pathway
  • load_ai_parameters()          — load AI LoRA weights
  • forward() now accepts activate_adv AND activate_ai flags
"""

import torch
import torch.nn as nn
import math

# ---------------------------------------------------------------------------
# Original ForgeryDetector (unchanged)
# ---------------------------------------------------------------------------

class ForgeryDetector(nn.Module):
    def __init__(self, input_channels=256, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, C, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, C)
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_map):
        """
        feat_map: tensor of shape (B, C, H, W), from SAM encoder
        returns: logits of shape (B, 2)
        """
        x = self.pool(feat_map)
        logits = self.classifier(x)
        return logits

# ---------------------------------------------------------------------------
# Original LoRA modules — shared forgery experts (unchanged)
# ---------------------------------------------------------------------------

class LoRA_QKV(nn.Module):
    def __init__(self, ori_qkv, dim, r):
        super().__init__()
        self.activate = True
        self.ori_qkv = ori_qkv
        self.dim = dim
        self.a_q = nn.Linear(dim, r, bias=False)
        self.b_q = nn.Linear(r, dim, bias=False)
        self.a_k = nn.Linear(dim, r, bias=False)
        self.b_k = nn.Linear(r, dim, bias=False)
        self.a_v = nn.Linear(dim, r, bias=False)
        self.b_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.kaiming_uniform_(self.a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_k.weight)
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v.weight)

    def set_shared_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        qkv = self.ori_qkv(x)  # [B,H,W,3C]
        if self.activate:
            qkv[:, :, :, : self.dim] +=self.b_q(self.a_q(x))
            qkv[:, :, :, self.dim:-self.dim] +=self.b_k(self.a_k(x))
            qkv[:, :, :, -self.dim:] += self.b_v(self.a_v(x))
        return qkv

class LoRA_FFN(nn.Module):
    def __init__(self, ori_lin, in_features, out_features, r):
        super().__init__()
        self.activate = True
        self.ori_lin = ori_lin
        self.a = nn.Linear(in_features, r, bias=False)
        self.b = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def set_shared_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        out = self.ori_lin(x)
        if self.activate:
            out = out + self.b(self.a(x))
        return out

# ---------------------------------------------------------------------------
# Original Adversarial LoRA modules (unchanged)
# ---------------------------------------------------------------------------

class Adv_LoRA_QKV(nn.Module):
    def __init__(self, ori_qkv, dim, r):
        super().__init__()
        self.activate = False
        self.ori_qkv = ori_qkv
        self.dim = dim
        self.a_q = nn.Linear(dim, r, bias=False)
        self.b_q = nn.Linear(r, dim, bias=False)
        self.a_k = nn.Linear(dim, r, bias=False)
        self.b_k = nn.Linear(r, dim, bias=False)
        self.a_v = nn.Linear(dim, r, bias=False)
        self.b_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.kaiming_uniform_(self.a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_k.weight)
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v.weight)

    def set_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        qkv = self.ori_qkv(x)  # [B,H,W,3C]
        # adversarial expert
        if self.activate:
            qkv[:, :, :, : self.dim] += self.b_q(self.a_q(x))
            qkv[:, :, :, self.dim:-self.dim] += self.b_k(self.a_k(x))
            qkv[:, :, :, -self.dim:] += self.b_v(self.a_v(x))

        return qkv

class Adv_LoRA_FFN(nn.Module):
    def __init__(self, ori_lin, in_features, out_features, r):
        super().__init__()
        self.activate = False
        self.ori_lin = ori_lin
        self.a = nn.Linear(in_features, r, bias=False)
        self.b = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def set_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        out = self.ori_lin(x)
        if self.activate:
            out = out + self.b(self.a(x))
        return out

# ---------------------------------------------------------------------------
# NEW: AI-Forgery LoRA modules
# ---------------------------------------------------------------------------

class AI_LoRA_QKV(nn.Module):
    """
    LoRA adapter for the QKV projection, fine-tuned specifically on
    AI-generated / diffusion-model forgery images (TGIF / TGIF2 dataset).

    Rank is intentionally larger (8*r) than the standard LoRA to capture
    the richer semantic inconsistencies introduced by generative models.

    activate=False by default; switched on only when AIForgeryClassifier
    predicts an AI-generated forgery.
    """
    def __init__(self, ori_qkv, dim, r):
        super().__init__()
        self.activate = False
        self.ori_qkv = ori_qkv
        self.dim = dim
        self.a_q = nn.Linear(dim, r, bias=False)
        self.b_q = nn.Linear(r, dim, bias=False)
        self.a_k = nn.Linear(dim, r, bias=False)
        self.b_k = nn.Linear(r, dim, bias=False)
        self.a_v = nn.Linear(dim, r, bias=False)
        self.b_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.kaiming_uniform_(self.a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_k.weight)
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v.weight)

    def set_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        qkv = self.ori_qkv(x)  # [B, H, W, 3C]
        if self.activate:
            qkv[:, :, :, : self.dim]           += self.b_q(self.a_q(x))
            qkv[:, :, :, self.dim:-self.dim]   += self.b_k(self.a_k(x))
            qkv[:, :, :, -self.dim:]           += self.b_v(self.a_v(x))
        return qkv


class AI_LoRA_FFN(nn.Module):
    """
    LoRA adapter for the first FFN linear layer, fine-tuned on AI-generated
    forgery images.  Mirrors AI_LoRA_QKV in design philosophy.
    """
    def __init__(self, ori_lin, in_features, out_features, r):
        super().__init__()
        self.activate = False
        self.ori_lin = ori_lin
        self.a = nn.Linear(in_features, r, bias=False)
        self.b = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def set_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        out = self.ori_lin(x)
        if self.activate:
            out = out + self.b(self.a(x))
        return out

# ---------------------------------------------------------------------------
# Main ForensicsSAM model — extended with AI forgery expert
# ---------------------------------------------------------------------------

class ForensicsSAM(nn.Module):
    """
    Extended ForensicsSAM model supporting three forgery expert pathways:

    1. Shared Forgery Experts  — always active, pre-trained, frozen
    2. Adversarial LoRA        — activated when adversarial perturbation detected
    3. AI-Forgery LoRA  (NEW)  — activated when AI-generated forgery detected

    Parameters
    ----------
    sam_model : SAM
        Base SAM model loaded from segment_anything registry.
    r : int
        LoRA rank for the shared and adversarial experts (default 8).
    lora_layer : list[int] | None
        Which transformer block indices to apply shared LoRA to.
        None → apply to all blocks.
    ai_lora_rank_multiplier : int
        Rank multiplier for the AI-forgery LoRA relative to r (default 8,
        giving rank = 8*r = 64 for r=8).
    load_ai_weights : bool
        Whether to attempt loading AI forgery weights on init.
        Set False during training before weights are saved.
    """

    def __init__(
        self,
        sam_model,
        r: int = 8,
        lora_layer=None,
        ai_lora_rank_multiplier: int = 8,
        load_ai_weights: bool = True,
    ):
        super().__init__()
        assert r > 0
        self.lora_layer = lora_layer or list(range(len(sam_model.image_encoder.blocks)))

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        blk_num = len(sam_model.image_encoder.blocks)
        if blk_num == 32:
            self.global_attn_index = [7, 15, 23, 31]
        elif blk_num == 24:
            self.global_attn_index = [5, 11, 17, 23]
        elif blk_num == 12:
            self.global_attn_index = [2, 5, 8, 11]

        # ---- Shared Forgery Experts ----
        # Cache dim/feature sizes from first applicable block for later use
        self._lora_dim = None
        self._lora_in_features = None
        self._lora_out_features = None

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            blk.attn.qkv = LoRA_QKV(w_qkv_linear, dim, r)

            w_lin1 = blk.mlp.lin1
            in_features = w_lin1.in_features
            out_features = w_lin1.out_features
            blk.mlp.lin1 = LoRA_FFN(w_lin1, in_features, out_features, r)

            # Cache sizes
            if self._lora_dim is None:
                self._lora_dim = dim
                self._lora_in_features = in_features
                self._lora_out_features = out_features

            # Freeze shared expert LoRA params
            for param in blk.attn.qkv.parameters():
                param.requires_grad = False
            for param in blk.mlp.lin1.parameters():
                param.requires_grad = False

        # Freeze prompt_encoder and mask_decoder
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        # ---- Forgery Detector ----
        self.detector = ForgeryDetector()
        for param in self.detector.parameters():
            param.requires_grad = False

        self.sam = sam_model
        self.load_all_parameters(f"./weight/forgery_experts.pth")

        # ---- Adaptive Adversarial Experts ----
        dim = self._lora_dim
        in_features = self._lora_in_features
        out_features = self._lora_out_features
        for blk in [self.sam.image_encoder.blocks[i] for i in self.global_attn_index]:
            w_qkv_linear = blk.attn.qkv
            blk.attn.qkv = Adv_LoRA_QKV(w_qkv_linear, dim, 8 * r)

            w_lin1 = blk.mlp.lin1
            blk.mlp.lin1 = Adv_LoRA_FFN(w_lin1, in_features, out_features, 8 * r)
        self.load_all_parameters(f"./weight/adversary_experts.pth")

        # ---- NEW: AI-Forgery Experts ----
        # Applied on top of the existing Adv_LoRA wrappers (chain: ori → Adv → AI)
        ai_rank = ai_lora_rank_multiplier * r
        for blk in [self.sam.image_encoder.blocks[i] for i in self.global_attn_index]:
            # The current qkv is an Adv_LoRA_QKV; wrap it again with AI_LoRA_QKV
            w_qkv_wrapped = blk.attn.qkv
            blk.attn.qkv = AI_LoRA_QKV(w_qkv_wrapped, dim, ai_rank)

            w_lin1_wrapped = blk.mlp.lin1
            blk.mlp.lin1 = AI_LoRA_FFN(w_lin1_wrapped, in_features, out_features, ai_rank)

        # Optionally load pre-trained AI forgery expert weights
        ai_weight_path = "./weight/ai_forgery_experts.pth"
        if load_ai_weights:
            import os
            if os.path.exists(ai_weight_path):
                self.load_ai_parameters(ai_weight_path)
                print(f"[ForensicsSAM] Loaded AI forgery experts from {ai_weight_path}")
            else:
                print(f"[ForensicsSAM] AI forgery weight not found at {ai_weight_path}. "
                      f"Run train_ai_lora.py to generate it.")

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------

    def activate_adv(self, activate):
        """Toggle adversarial expert LoRA."""
        for blk in [self.sam.image_encoder.blocks[i] for i in self.global_attn_index]:
            # qkv is now AI_LoRA_QKV wrapping Adv_LoRA_QKV
            blk.attn.qkv.ori_qkv.set_activate(activate)
            blk.mlp.lin1.ori_lin.set_activate(activate)

    def activate_ai(self, activate):
        """Toggle AI-forgery expert LoRA."""
        for blk in [self.sam.image_encoder.blocks[i] for i in self.global_attn_index]:
            blk.attn.qkv.set_activate(activate)
            blk.mlp.lin1.set_activate(activate)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images, activate_adv, activate_ai=False):
        """
        Parameters
        ----------
        images : torch.Tensor  (B, 3, H, W)
        activate_adv : bool | torch.Tensor
            Whether to activate the adversarial expert LoRA.
        activate_ai : bool | torch.Tensor
            Whether to activate the AI-forgery expert LoRA.
            When a Tensor (batch predictions from AIForgeryClassifier), the
            first item is used to gate activation (single-image inference).

        Returns
        -------
        mask_prediction : torch.Tensor  (B, 1, H//4, W//4)
        cls_prediction  : torch.Tensor  (B, 1)
        """
        # Resolve tensor predictions to scalar bool for routing
        if isinstance(activate_ai, torch.Tensor):
            activate_ai = bool(activate_ai.any().item())
        if isinstance(activate_adv, torch.Tensor):
            activate_adv = bool(activate_adv.any().item())

        self.activate_adv(activate_adv)
        self.activate_ai(activate_ai)

        image_embeddings = self.sam.image_encoder(images)
        cls_prediction = self.detector(image_embeddings)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        device = image_embeddings.device
        mask_prediction, iou_prediction = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe().to(device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )  # (1, 1, H//4, W//4)  (1, 1)
        return mask_prediction, cls_prediction

    # ------------------------------------------------------------------
    # Parameter I/O helpers (original, unchanged)
    # ------------------------------------------------------------------

    def load_sam_lora_parameters(self, path):
        state_dict = torch.load(path, map_location='cpu')
        self._load_state_dict(self.sam, state_dict)

    def load_detector_parameters(self, path):
        if self.detector is None:
            raise ValueError("Detector is not defined.")
        state_dict = torch.load(path, map_location='cpu')
        self._load_state_dict(self.detector, state_dict)

    def save_all_parameters(self, path):
        all_params = {}
        all_params["sam"] = {
            k: v for k, v in self._get_named_params(self.sam) if v.requires_grad
        }
        if self.detector is not None:
            all_params["detector"] = {
                k: v for k, v in self._get_named_params(self.detector) if v.requires_grad
            }
        torch.save(all_params, path)

    def load_all_parameters(self, path):
        all_params = torch.load(path, map_location='cpu')
        print(path)
        if "sam" in all_params:
            self._load_state_dict(self.sam, all_params["sam"])
        if "detector" in all_params and self.detector is not None:
            self._load_state_dict(self.detector, all_params["detector"])

    # ------------------------------------------------------------------
    # NEW: AI-Forgery expert parameter I/O
    # ------------------------------------------------------------------

    def save_ai_parameters(self, path: str):
        """Save only the trainable AI-forgery LoRA parameters."""
        ai_params = {}
        for name, param in self.sam.named_parameters():
            # AI LoRA params are on the outermost LoRA wrapper (AI_LoRA_*)
            # They appear in names like:
            #   image_encoder.blocks.7.attn.qkv.a_q.weight  (but only if
            #   the AI wrapper is the outermost one)
            if param.requires_grad:
                ai_params[name] = param
        torch.save({"sam": ai_params}, path)
        print(f"[ForensicsSAM] AI forgery experts saved to {path}")

    def load_ai_parameters(self, path: str):
        """Load AI-forgery LoRA parameters."""
        all_params = torch.load(path, map_location='cpu')
        if "sam" in all_params:
            self._load_state_dict(self.sam, all_params["sam"])

    def set_ai_lora_trainable(self, trainable: bool = True):
        """
        Make AI-forgery LoRA params (and only those) trainable/frozen.
        Call this before running train_ai_lora.py.
        """
        for blk in [self.sam.image_encoder.blocks[i] for i in self.global_attn_index]:
            for param in blk.attn.qkv.parameters():
                param.requires_grad = False  # reset all first
            for param in blk.mlp.lin1.parameters():
                param.requires_grad = False
            # Then re-enable only the outermost (AI) adapter params
            if trainable:
                for pname in ["a_q", "b_q", "a_k", "b_k", "a_v", "b_v"]:
                    for param in getattr(blk.attn.qkv, pname).parameters():
                        param.requires_grad = True
                for pname in ["a", "b"]:
                    for param in getattr(blk.mlp.lin1, pname).parameters():
                        param.requires_grad = True

    # ------------------------------------------------------------------
    # Internal utilities (unchanged)
    # ------------------------------------------------------------------

    def _get_named_params(self, module):
        if isinstance(module, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            return module.module.named_parameters()
        return module.named_parameters()

    def _load_state_dict(self, module, state_dict):
        if isinstance(module, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            module.module.load_state_dict(state_dict, strict=False)
        else:
            module.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Registry (unchanged)
# ---------------------------------------------------------------------------

model_type = ['vit_b', 'vit_l', 'vit_h']
checkpoint = {
    'vit_b': './weight/pre-trained_weight/sam_vit_b_01ec64.pth',
    'vit_l': './weight/pre-trained_weight/sam_vit_l_0b3195.pth',
    'vit_h': './weight/pre-trained_weight/sam_vit_h_4b8939.pth'
}
