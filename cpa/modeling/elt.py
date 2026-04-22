from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# ELT-DiT model
# =============================================================================


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, input_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by patch_size ({patch_size}).")
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        use_cfg_token = 1 if dropout_prob > 0 else 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_token, hidden_size)

    def token_drop(
        self, labels: torch.Tensor, force_drop_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if force_drop_mask is None:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop = force_drop_mask.to(dtype=torch.bool)
        null_label = torch.full_like(labels, self.num_classes)
        return torch.where(drop, null_label, labels)

    def forward(
        self, labels: torch.Tensor, train: bool, force_drop_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        do_dropout = self.dropout_prob > 0 and (train or force_drop_mask is not None)
        if do_dropout:
            labels = self.token_drop(labels, force_drop_mask=force_drop_mask)
        return self.embedding_table(labels)


class ELTDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadSelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            6, dim=1
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


@dataclass
class ELTDiTConfig:
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    num_classes: int = 1000
    hidden_size: int = 2048
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_unique_layers: int = 8
    max_loops: int = 4
    class_dropout_prob: float = 0.10
    out_channels: Optional[int] = None  # None => v-pred with same channels as input

    @property
    def effective_depth(self) -> int:
        return self.num_unique_layers * self.max_loops


class ELTDiT(nn.Module):
    """
    Diffusion-side ELT implementation.

    This matches the paper's core idea exactly:
      - a composite block g_Theta made of N unique DiT blocks
      - recursive reuse of g_Theta for L loops within each denoising step
      - a shared final head so the model can exit at any loop count

    It also exposes the teacher/student prefix capture used by ILSD.
    """

    def __init__(self, cfg: ELTDiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_channels = cfg.in_channels if cfg.out_channels is None else cfg.out_channels

        self.x_embedder = PatchEmbed(cfg.input_size, cfg.patch_size, cfg.in_channels, cfg.hidden_size)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.y_embedder = LabelEmbedder(cfg.num_classes, cfg.hidden_size, cfg.class_dropout_prob)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, cfg.hidden_size),
            requires_grad=False,
        )
        self.unique_blocks = nn.ModuleList(
            [
                ELTDiTBlock(cfg.hidden_size, cfg.num_heads, mlp_ratio=cfg.mlp_ratio)
                for _ in range(cfg.num_unique_layers)
            ]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.unique_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        p = self.cfg.patch_size
        h = w = int(t**0.5)
        if h * w != t:
            raise ValueError(f"Token count {t} is not a square; cannot unpatchify.")
        x = x.view(b, h, w, p, p, self.out_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(b, self.out_channels, h * p, w * p)

    def embed_inputs(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        force_drop_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tokens = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, train=self.training, force_drop_mask=force_drop_mask)
        return x_tokens, c

    def run_loops(
        self,
        x_tokens: torch.Tensor,
        c: torch.Tensor,
        num_loops: int,
        capture_after_loop: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        if capture_after_loop is not None and not (1 <= capture_after_loop <= num_loops):
            raise ValueError("capture_after_loop must lie in [1, num_loops]")

        captured = None
        x_curr = x_tokens
        for loop_idx in range(num_loops):
            for block in self.unique_blocks:
                x_curr = block(x_curr, c)
            if capture_after_loop is not None and (loop_idx + 1) == capture_after_loop:
                captured = x_curr
        return x_curr, captured

    def decode_tokens(self, tokens: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.unpatchify(self.final_layer(tokens, c))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        num_loops: Optional[int] = None,
        force_drop_mask: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        loops = self.cfg.max_loops if num_loops is None else num_loops
        x_tokens, c = self.embed_inputs(x, t, y, force_drop_mask=force_drop_mask)
        tokens, _ = self.run_loops(x_tokens, c, loops)
        out = self.decode_tokens(tokens, c)
        if return_tokens:
            return out, tokens
        return out

    def forward_teacher_student(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        student_loops: int,
        force_drop_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Single-pass prefix capture for ILSD:
          - student state at L_int
          - teacher state at L_max

        This follows the paper's key efficiency property that the student path is a strict prefix
        of the teacher path, so the teacher does not need a second forward pass.
        """
        if not (1 <= student_loops < self.cfg.max_loops):
            raise ValueError(f"student_loops must be in [1, {self.cfg.max_loops - 1}]")

        x_tokens, c = self.embed_inputs(x, t, y, force_drop_mask=force_drop_mask)
        teacher_tokens, student_tokens = self.run_loops(
            x_tokens,
            c,
            num_loops=self.cfg.max_loops,
            capture_after_loop=student_loops,
        )
        assert student_tokens is not None

        return {
            "student_tokens": student_tokens,
            "teacher_tokens": teacher_tokens,
            "student_pred": self.decode_tokens(student_tokens, c),
            "teacher_pred": self.decode_tokens(teacher_tokens, c),
            "student_loops": torch.tensor(student_loops, device=x.device),
        }

    @torch.no_grad()
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        num_loops: Optional[int] = None,
    ) -> torch.Tensor:
        if cfg_scale == 1.0:
            return self(x, t, y, num_loops=num_loops)

        uncond_y = torch.full_like(y, self.cfg.num_classes)
        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        y_cat = torch.cat([y, uncond_y], dim=0)
        force_drop = torch.cat(
            [torch.zeros_like(y, dtype=torch.bool), torch.ones_like(y, dtype=torch.bool)],
            dim=0,
        )
        model_out = self(
            x_cat,
            t_cat,
            y_cat,
            num_loops=num_loops,
            force_drop_mask=force_drop,
        )
        cond, uncond = model_out.chunk(2, dim=0)
        return uncond + cfg_scale * (cond - uncond)


# =============================================================================
# Diffusion utilities
# =============================================================================


@dataclass
class ShiftedCosineScheduleConfig:
    """
    ELT says it uses a shifted cosine schedule, but the uploaded manuscript excerpt does not give the
    exact latent-space shift constant. So the implementation exposes it explicitly through either:
      - (image_d, noise_d), matching simple-diffusion's shifted log-SNR formula, or
      - direct_logsnr_shift.
    """

    num_steps: int = 512
    logsnr_min: float = -15.0
    logsnr_max: float = 15.0
    image_d: float = 32.0
    noise_d: float = 32.0
    direct_logsnr_shift: Optional[float] = None
    max_beta: float = 0.999


class ShiftedCosineSchedule:
    def __init__(self, cfg: ShiftedCosineScheduleConfig) -> None:
        self.cfg = cfg
        self.num_steps = cfg.num_steps

        betas = self._make_betas().float()
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        self.posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

    def to(self, device: torch.device | str) -> "ShiftedCosineSchedule":
        for name in [
            "betas",
            "alphas",
            "alpha_bars",
            "alpha_bars_prev",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _base_logsnr_cosine(self, t: torch.Tensor) -> torch.Tensor:
        t_min = math.atan(math.exp(-0.5 * self.cfg.logsnr_max))
        t_max = math.atan(math.exp(-0.5 * self.cfg.logsnr_min))
        return -2.0 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr(self, t: torch.Tensor) -> torch.Tensor:
        base = self._base_logsnr_cosine(t)
        if self.cfg.direct_logsnr_shift is not None:
            shift = self.cfg.direct_logsnr_shift
        else:
            shift = 2.0 * math.log(self.cfg.noise_d / self.cfg.image_d)
        return base + shift

    def alpha_bar_continuous(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logsnr(t))

    def _make_betas(self) -> torch.Tensor:
        betas = []
        for i in range(self.num_steps):
            t1 = torch.tensor(i / self.num_steps, dtype=torch.float64)
            t2 = torch.tensor((i + 1) / self.num_steps, dtype=torch.float64)
            alpha_bar_t1 = self.alpha_bar_continuous(t1)
            alpha_bar_t2 = self.alpha_bar_continuous(t2)
            beta = 1.0 - alpha_bar_t2 / alpha_bar_t1
            betas.append(min(float(beta), self.cfg.max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device)

    def continuous_t_from_index(self, t_idx: torch.Tensor) -> torch.Tensor:
        return (t_idx.float() + 0.5) / self.num_steps

    @staticmethod
    def _extract(arr: torch.Tensor, t_idx: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = arr.gather(0, t_idx)
        while out.ndim < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(
        self, x0: torch.Tensor, t_idx: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t_idx, x0.shape)
        sqrt_omb = self._extract(self.sqrt_one_minus_alpha_bars, t_idx, x0.shape)
        xt = sqrt_ab * x0 + sqrt_omb * noise
        return xt, noise

    def v_target(self, x0: torch.Tensor, eps: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alpha_bars, t_idx, x0.shape)
        sigma = self._extract(self.sqrt_one_minus_alpha_bars, t_idx, x0.shape)
        return alpha * eps - sigma * x0

    def predict_x0_from_v(self, xt: torch.Tensor, v: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alpha_bars, t_idx, xt.shape)
        sigma = self._extract(self.sqrt_one_minus_alpha_bars, t_idx, xt.shape)
        return alpha * xt - sigma * v

    def predict_eps_from_v(self, xt: torch.Tensor, v: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alpha_bars, t_idx, xt.shape)
        sigma = self._extract(self.sqrt_one_minus_alpha_bars, t_idx, xt.shape)
        return sigma * xt + alpha * v

    def q_posterior_mean_variance(
        self, x0_pred: torch.Tensor, xt: torch.Tensor, t_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            self._extract(self.posterior_mean_coef1, t_idx, xt.shape) * x0_pred
            + self._extract(self.posterior_mean_coef2, t_idx, xt.shape) * xt
        )
        var = self._extract(self.posterior_variance, t_idx, xt.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t_idx, xt.shape)
        return mean, var, log_var


@dataclass
class ILSDLossConfig:
    min_student_loops: int = 1
    lambda_start: float = 1.0
    lambda_end: float = 0.0
    sigmoid_bias: float = -3.0
    distill_on: str = "features"  # "features" matches the ELT algorithm box


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    loss = (pred - target) ** 2
    if weight is not None:
        loss = loss * weight
    return loss.flatten(1).mean(dim=1).mean()


class ELTDiffusionTrainer:
    """
    Single-pass ELT + ILSD training step for the diffusion branch.

    Important note:
    - The algorithm box in ELT computes distillation on F_int and stopgrad(F_max), i.e. feature-space
      distillation before the shared head. That is the default here via distill_on="features".
    """

    def __init__(self, model: ELTDiT, schedule: ShiftedCosineSchedule, cfg: ILSDLossConfig) -> None:
        self.model = model
        self.schedule = schedule
        self.cfg = cfg

    def lambda_at_progress(self, progress: float) -> float:
        progress = float(max(0.0, min(1.0, progress)))
        return self.cfg.lambda_start + (self.cfg.lambda_end - self.cfg.lambda_start) * progress

    def sigmoid_loss_weight(self, t_continuous: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid-weighted ELBO-style MSE from the SiD2 appendix:

            w(t) = -0.5 * d logSNR(t) / dt * exp(bias) * sigmoid(logSNR(t) - bias)

        ELT cites the simpler-diffusion family for this weighting, but the uploaded TeX excerpt does
        not expose the exact chosen bias; this implementation makes it explicit and configurable.
        """
        t = t_continuous.detach().requires_grad_(True)
        logsnr = self.schedule.logsnr(t)
        dlogsnr_dt = torch.autograd.grad(logsnr.sum(), t, create_graph=False)[0]
        weight = (
            -0.5
            * dlogsnr_dt
            * math.exp(self.cfg.sigmoid_bias)
            * torch.sigmoid(logsnr - self.cfg.sigmoid_bias)
        )
        return weight.detach()

    def sample_student_loops(self, device: torch.device) -> int:
        low = self.cfg.min_student_loops
        high = self.model.cfg.max_loops
        if not (1 <= low < high):
            raise ValueError(f"min_student_loops must be in [1, {high - 1}]")
        return int(torch.randint(low=low, high=high, size=(1,), device=device).item())

    def training_step(
        self,
        x0: torch.Tensor,
        class_labels: torch.Tensor,
        progress: float,
        t_idx: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        student_loops: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        device = x0.device
        bsz = x0.shape[0]

        if t_idx is None:
            t_idx = self.schedule.sample_timesteps(batch_size=bsz, device=device)
        if noise is None:
            noise = torch.randn_like(x0)
        if student_loops is None:
            student_loops = self.sample_student_loops(device=device)

        xt, eps = self.schedule.q_sample(x0, t_idx, noise=noise)
        v_target = self.schedule.v_target(x0, eps, t_idx)
        t_cont = self.schedule.continuous_t_from_index(t_idx)
        t_embed = t_idx.float()

        lam = self.lambda_at_progress(progress)
        w = self.sigmoid_loss_weight(t_cont)
        while w.ndim < x0.ndim:
            w = w.unsqueeze(-1)

        outputs = self.model.forward_teacher_student(
            xt,
            t_embed,
            class_labels,
            student_loops=student_loops,
        )

        teacher_gt = weighted_mse(outputs["teacher_pred"], v_target, weight=w)
        student_gt = weighted_mse(outputs["student_pred"], v_target, weight=w)

        if self.cfg.distill_on == "features":
            distill = weighted_mse(outputs["student_tokens"], outputs["teacher_tokens"].detach(), weight=None)
        elif self.cfg.distill_on == "outputs":
            distill = weighted_mse(outputs["student_pred"], outputs["teacher_pred"].detach(), weight=w)
        else:
            raise ValueError(f"Unknown distill_on={self.cfg.distill_on!r}")

        total = teacher_gt + lam * student_gt + (1.0 - lam) * distill
        return {
            "loss": total,
            "teacher_gt": teacher_gt.detach(),
            "student_gt": student_gt.detach(),
            "distill": distill.detach(),
            "lambda": torch.tensor(lam, device=device),
            "student_loops": torch.tensor(student_loops, device=device),
            "t_idx": t_idx.detach(),
        }


@torch.no_grad()
def sample_ddpm(
    model: ELTDiT,
    schedule: ShiftedCosineSchedule,
    shape: Sequence[int],
    class_labels: torch.Tensor,
    loop_budget: int | Sequence[int],
    cfg_scale: float = 3.0,
    device: Optional[torch.device] = None,
    clip_x0: Optional[float] = 1.0,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    x = torch.randn(*shape, device=device)
    class_labels = class_labels.to(device)
    batch = x.shape[0]

    if isinstance(loop_budget, int):
        per_step_loops = [loop_budget] * schedule.num_steps
    else:
        per_step_loops = list(loop_budget)
        if len(per_step_loops) != schedule.num_steps:
            raise ValueError("If loop_budget is a sequence, it must have length == num_steps.")

    for step in reversed(range(schedule.num_steps)):
        t_idx = torch.full((batch,), step, device=device, dtype=torch.long)
        v_pred = model.forward_with_cfg(
            x,
            t_idx.float(),
            class_labels,
            cfg_scale=cfg_scale,
            num_loops=int(per_step_loops[step]),
        )
        x0_pred = schedule.predict_x0_from_v(x, v_pred, t_idx)
        if clip_x0 is not None:
            x0_pred = x0_pred.clamp(-clip_x0, clip_x0)

        mean, _, log_var = schedule.q_posterior_mean_variance(x0_pred, x, t_idx)
        if step > 0:
            x = mean + torch.exp(0.5 * log_var) * torch.randn_like(x)
        else:
            x = mean
    return x


# =============================================================================
# Positional embeddings + presets
# =============================================================================


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for 2D sin-cos positional embeddings.")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for 1D sin-cos positional embeddings.")
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1).to(torch.float64)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).to(torch.float32)


def elt_dit_paper_large_8x4() -> ELTDiT:
    return ELTDiT(
        ELTDiTConfig(
            hidden_size=2048,
            num_heads=16,
            num_unique_layers=8,
            max_loops=4,
            input_size=32,
            patch_size=2,
            in_channels=4,
            num_classes=1000,
        )
    )


def elt_dit_paper_large_16x2() -> ELTDiT:
    return ELTDiT(
        ELTDiTConfig(
            hidden_size=2048,
            num_heads=16,
            num_unique_layers=16,
            max_loops=2,
            input_size=32,
            patch_size=2,
            in_channels=4,
            num_classes=1000,
        )
    )


def elt_dit_toy() -> ELTDiT:
    return ELTDiT(
        ELTDiTConfig(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=128,
            num_heads=4,
            num_unique_layers=2,
            max_loops=3,
            num_classes=10,
        )
    )


if __name__ == "__main__":
    # tiny smoke test on CPU
    device = torch.device("mps")
    model = elt_dit_toy().to(device)
    schedule = ShiftedCosineSchedule(ShiftedCosineScheduleConfig(num_steps=2)).to(device)
    trainer = ELTDiffusionTrainer(model, schedule, ILSDLossConfig(min_student_loops=1, sigmoid_bias=-3.0))

    x0 = torch.randn(2, 4, 8, 8, device=device)
    y = torch.randint(0, 10, (2,), device=device)

    out = model.forward_teacher_student(x0, torch.zeros(2, device=device), y, student_loops=1)
    print("teacher_pred", tuple(out["teacher_pred"].shape))
    print("student_pred", tuple(out["student_pred"].shape))

    losses = trainer.training_step(
        x0=x0,
        class_labels=y,
        progress=0.5,
        t_idx=torch.zeros(2, device=device, dtype=torch.long),
        student_loops=1,
    )
    print("loss", float(losses["loss"].detach()))
