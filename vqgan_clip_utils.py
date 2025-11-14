"""
Modern VQGAN+CLIP Utilities (2025)
Based on rkhamilton architecture with modernized dependencies and techniques.
"""

import sys
from pathlib import Path

# Add taming-transformers to path - check common locations
# Priority: /mnt/store (persistent), parent dir (local), relative path
taming_locations = [
    Path("/mnt/store/taming-transformers"),  # Persistent notebook storage
    Path(__file__).parent / "taming-transformers",  # Same directory as this file
]

for taming_path in taming_locations:
    if taming_path.exists() and str(taming_path) not in sys.path:
        sys.path.insert(0, str(taming_path))
        break  # Use first found location

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torchvision.transforms import functional as TF
try:
    import open_clip
except ImportError:
    # Package is installed as open-clip-torch but can be imported as open_clip_torch
    import open_clip_torch as open_clip
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from pathlib import Path
import kornia.augmentation as K
from typing import List, Optional, Tuple, Dict, Any
import sys


# ============================================================================
# Device Configuration
# ============================================================================

def get_device(device_type: str = "auto", gpu_id: int = 0) -> Tuple[torch.device, torch.dtype]:
    """
    Auto-detect optimal device and dtype for mixed precision.

    Args:
        device_type: "auto", "cuda", "mps", or "cpu"
        gpu_id: GPU index for multi-GPU systems

    Returns:
        (device, dtype) tuple
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            # Check for A100/H100 native bfloat16 support
            gpu_name = torch.cuda.get_device_name(gpu_id).lower()
            if "a100" in gpu_name or "h100" in gpu_name:
                dtype = torch.bfloat16
                print(f"Using CUDA device: {gpu_name} with native bfloat16")
            else:
                dtype = torch.float16
                print(f"Using CUDA device: {gpu_name} with float16")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32  # MPS doesn't support float16/bfloat16 well
            print("Using Apple Silicon (MPS) with float32")
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            print("Using CPU with float32")
    else:
        device = torch.device(device_type)
        if device_type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

    return device, dtype


# ============================================================================
# Model Loading
# ============================================================================

def load_vqgan(config_path: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load VQGAN model from config and checkpoint.

    Args:
        config_path: Path to VQGAN config YAML
        checkpoint_path: Path to VQGAN checkpoint
        device: Target device

    Returns:
        VQGAN model ready for inference
    """
    # Load config
    config = OmegaConf.load(config_path)

    # Import VQGAN model class
    # Note: This requires taming-transformers to be installed
    try:
        from taming.models.vqgan import VQModel, GumbelVQ
    except ImportError:
        raise ImportError(
            "taming-transformers not found. Install with:\n"
            "pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
        )

    # Determine model class
    model_type = config.model.target.split('.')[-1]
    if model_type == "VQModel":
        model = VQModel(**config.model.params)
    elif model_type == "GumbelVQ":
        model = GumbelVQ(**config.model.params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    # Note: weights_only=False is required for PyTorch 2.6+ to load PyTorch Lightning checkpoints
    # Only use checkpoints from trusted sources (official VQGAN models)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.eval()

    # MPS compatibility: keep VQGAN on CPU due to unsupported padding modes
    if device.type == "mps":
        print("Applying MPS compatibility wrapper for VQGAN (model kept on CPU)...")
        model = _wrap_vqgan_for_mps(model)
    else:
        model = model.to(device)

    # Disable gradients for VQGAN
    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded VQGAN from {checkpoint_path}")
    return model


def _wrap_vqgan_for_mps(model: nn.Module) -> nn.Module:
    """
    Wrap VQGAN model to be MPS-compatible.
    MPS doesn't support some padding modes, so we keep model on CPU
    and handle device transfers in decode/encode.
    """
    # Keep the model on CPU to avoid MPS padding issues
    model = model.cpu()

    original_decode = model.decode
    original_encode = model.encode

    def mps_safe_decode(z):
        """Decode on CPU, input comes from MPS - preserves gradients"""
        original_device = z.device
        z_cpu = z.cpu()
        # Don't use no_grad here - we need gradients to flow back to z
        result = original_decode(z_cpu)
        return result.to(original_device)

    def mps_safe_encode(x):
        """Encode on CPU, input comes from MPS"""
        original_device = x.device
        x_cpu = x.cpu()
        with torch.no_grad():
            z, _, info = original_encode(x_cpu)
        return z.to(original_device), None, info

    # Replace methods with MPS-safe versions
    model.decode = mps_safe_decode
    model.encode = mps_safe_encode

    return model


def load_clip(model_name: str = "ViT-B-32",
              pretrained: str = "laion2b_s34b_b79k",
              device: torch.device = None) -> Tuple[nn.Module, Any, Any]:
    """
    Load OpenCLIP model.

    Args:
        model_name: OpenCLIP model name (e.g., "ViT-B-32", "ViT-L-14")
        pretrained: Pretrained weights dataset
        device: Target device

    Returns:
        (model, preprocess, tokenizer) tuple
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded OpenCLIP {model_name} with {pretrained} weights")
    return model, preprocess, tokenizer


# ============================================================================
# Cutout Generation (Kornia-based, extensible for custom patterns)
# ============================================================================

class MakeCutoutsKornia(nn.Module):
    """
    Kornia-based cutout generation with augmentations.
    This class is designed to be easily extended with custom sampling patterns.
    """

    def __init__(self,
                 cut_size: int = 224,
                 num_cuts: int = 32,
                 cut_power: float = 1.0,
                 affine_prob: float = 0.5,
                 jitter_prob: float = 0.5,
                 noise_prob: float = 0.1,
                 use_mps: bool = False):
        """
        Args:
            cut_size: Size to resize cutouts to (CLIP input size)
            num_cuts: Number of cutouts to generate
            cut_power: Distribution of cutout sizes (<1.0 = larger, >1.0 = smaller)
            affine_prob: Probability of affine transformations
            jitter_prob: Probability of color jitter
            noise_prob: Probability of adding noise
            use_mps: If True, use MPS-compatible padding mode (zeros instead of border)
        """
        super().__init__()
        self.cut_size = cut_size
        self.num_cuts = num_cuts
        self.cut_power = cut_power

        # MPS doesn't support 'border' padding mode, use 'zeros' instead
        padding_mode = 'zeros' if use_mps else 'border'

        # Augmentation pipeline
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=15, translate=0.1, p=affine_prob, padding_mode=padding_mode),
            K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=jitter_prob),
            K.RandomGrayscale(p=0.1),
            K.RandomGaussianNoise(p=noise_prob),
        )

        # Normalization for CLIP (ImageNet stats)
        self.normalize = K.Normalize(
            mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
            std=torch.tensor([0.26862954, 0.26130258, 0.27577711])
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate cutouts from image.

        Args:
            image: Input image tensor [B, C, H, W], values in [-1, 1]

        Returns:
            Cutouts tensor [num_cuts, C, cut_size, cut_size]
        """
        batch_size, _, h, w = image.shape

        # Convert from [-1, 1] to [0, 1] for processing
        image = (image + 1) / 2

        cutouts = []

        for _ in range(self.num_cuts):
            # Sample cutout size using power law distribution
            # This creates more variation in cutout sizes
            size_ratio = torch.rand(1) ** self.cut_power
            size = int(min(h, w) * (0.2 + 0.8 * size_ratio))

            # Random crop
            offsetx = torch.randint(0, w - size + 1, (1,)).item() if w > size else 0
            offsety = torch.randint(0, h - size + 1, (1,)).item() if h > size else 0

            cutout = image[:, :, offsety:offsety + size, offsetx:offsetx + size]

            # Resize to CLIP input size
            cutout = F.interpolate(cutout, size=(self.cut_size, self.cut_size),
                                  mode='bilinear', align_corners=False)

            cutouts.append(cutout)

        # Stack all cutouts
        cutouts = torch.cat(cutouts, dim=0)

        # Apply augmentations
        cutouts = self.augs(cutouts)

        # Normalize for CLIP
        cutouts = self.normalize(cutouts)

        return cutouts


class MakeCutoutsOriginal(nn.Module):
    """
    Original Katherine Crowson cutout method (simpler, faster).
    Kept for compatibility and as an alternative to Kornia method.
    """

    def __init__(self, cut_size: int = 224, num_cuts: int = 32, cut_power: float = 1.0):
        super().__init__()
        self.cut_size = cut_size
        self.num_cuts = num_cuts
        self.cut_power = cut_power

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = image.shape
        image = (image + 1) / 2  # [-1, 1] to [0, 1]

        cutouts = []

        for _ in range(self.num_cuts):
            size_ratio = torch.rand(1) ** self.cut_power
            size = int(min(h, w) * (0.2 + 0.8 * size_ratio))

            offsetx = torch.randint(0, w - size + 1, (1,)).item() if w > size else 0
            offsety = torch.randint(0, h - size + 1, (1,)).item() if h > size else 0

            cutout = image[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.interpolate(cutout, size=(self.cut_size, self.cut_size),
                                  mode='bilinear', align_corners=False)

            cutouts.append(cutout)

        cutouts = torch.cat(cutouts, dim=0)
        cutouts = self.normalize(cutouts)

        return cutouts


# ============================================================================
# Prompt Processing
# ============================================================================

def parse_prompt(prompt: str) -> Tuple[str, float]:
    """
    Parse prompt with optional weight.
    Format: "text:weight" or just "text" (default weight 1.0)

    Args:
        prompt: Prompt string

    Returns:
        (text, weight) tuple
    """
    parts = prompt.rsplit(":", 1)
    if len(parts) == 2 and parts[1].replace(".", "").replace("-", "").isdigit():
        return parts[0], float(parts[1])
    return prompt, 1.0


def encode_prompts(prompts: List[str],
                   clip_model: nn.Module,
                   tokenizer: Any,
                   device: torch.device) -> torch.Tensor:
    """
    Encode text prompts with CLIP.

    Args:
        prompts: List of text prompts (can include weights like "text:1.5")
        clip_model: OpenCLIP model
        tokenizer: OpenCLIP tokenizer
        device: Target device

    Returns:
        Weighted and normalized text embeddings
    """
    if not prompts:
        return None

    texts = []
    weights = []

    for prompt in prompts:
        text, weight = parse_prompt(prompt)
        texts.append(text)
        weights.append(weight)

    # Tokenize
    text_tokens = tokenizer(texts).to(device)

    # Encode
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Apply weights
    weights = torch.tensor(weights, device=device).unsqueeze(-1)
    text_features = text_features * weights

    # Normalize again after weighting
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


# ============================================================================
# Loss Calculation
# ============================================================================

def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate spherical distance between normalized vectors.
    More stable than cosine similarity for optimization.
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def calculate_clip_loss(image_features: torch.Tensor,
                       text_features: torch.Tensor,
                       use_spherical: bool = False) -> torch.Tensor:
    """
    Calculate CLIP loss between image and text features.

    Args:
        image_features: CLIP image embeddings [num_cutouts, embed_dim]
        text_features: CLIP text embeddings [num_prompts, embed_dim]
        use_spherical: Use spherical distance (recommended) vs cosine similarity

    Returns:
        Loss value (lower = better alignment)
    """
    # Normalize features (keeping gradients)
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity matrix: [num_cutouts, num_prompts]
    similarities = image_features_norm @ text_features_norm.T

    # We want to maximize similarity, so minimize negative similarity
    # Average across all cutout-prompt pairs
    return -similarities.mean()


# ============================================================================
# Optimization Loop
# ============================================================================

class VQGANCLIPOptimizer:
    """
    Main optimization loop for VQGAN+CLIP generation.
    """

    def __init__(self,
                 vqgan_model: nn.Module,
                 clip_model: nn.Module,
                 text_features: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype,
                 make_cutouts: nn.Module,
                 learning_rate: float = 0.1,
                 weight_decay: float = 0.01,
                 use_scheduler: bool = True,
                 iterations: int = 500):
        """
        Args:
            vqgan_model: VQGAN model
            clip_model: OpenCLIP model
            text_features: Encoded text prompts
            device: Compute device
            dtype: Mixed precision dtype
            make_cutouts: Cutout generation module
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW
            use_scheduler: Use cosine annealing scheduler
            iterations: Total number of iterations
        """
        self.vqgan = vqgan_model
        self.clip = clip_model
        self.text_features = text_features
        self.device = device
        self.dtype = dtype
        self.make_cutouts = make_cutouts
        self.iterations = iterations

        # Will be set when initializing latent
        self.z = None
        self.optimizer = None
        self.scheduler = None

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler

        # Mixed precision scaler (only for CUDA with float16/bfloat16)
        self.use_amp = device.type == "cuda" and dtype in [torch.float16, torch.bfloat16]
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    def initialize_latent(self,
                         size: Tuple[int, int],
                         init_image: Optional[str] = None,
                         init_weight: float = 0.0,
                         seed: Optional[int] = None) -> torch.Tensor:
        """
        Initialize latent vector.

        Args:
            size: (width, height) of output image
            init_image: Optional path to initialization image
            init_weight: Weight for init image loss (0.0-1.0)
            seed: Random seed for reproducibility

        Returns:
            Initialized latent tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Calculate latent dimensions
        f = 2 ** (self.vqgan.decoder.num_resolutions - 1)
        toksX, toksY = size[0] // f, size[1] // f

        if init_image is not None:
            # Initialize from image
            pil_image = Image.open(init_image).convert('RGB')
            pil_image = pil_image.resize(size, Image.LANCZOS)

            image_tensor = TF.to_tensor(pil_image).to(self.device)
            image_tensor = image_tensor * 2 - 1  # [0, 1] to [-1, 1]
            image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                z, _, _ = self.vqgan.encode(image_tensor)

            self.z_init = z.clone() if init_weight > 0 else None
            self.init_weight = init_weight
        else:
            # Initialize from noise
            z = torch.randn([1, self.vqgan.quantize.e_dim, toksY, toksX], device=self.device)
            self.z_init = None
            self.init_weight = 0.0

        self.z = z.requires_grad_(True)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            [self.z],
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # Setup scheduler
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.iterations,
                eta_min=self.lr * 0.01
            )

        return self.z

    def step(self, iteration: int) -> Tuple[torch.Tensor, float]:
        """
        Perform one optimization step.

        Args:
            iteration: Current iteration number

        Returns:
            (decoded_image, loss_value) tuple
        """
        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                loss, out = self._forward()

            # Backward with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, out = self._forward()
            loss.backward()
            self.optimizer.step()

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        # MPS memory management - clear cache periodically to prevent crashes
        if self.device.type == "mps" and iteration % 10 == 0:
            import gc
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        return out, loss.item()

    def _forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: decode, create cutouts, calculate loss.

        Returns:
            (loss, decoded_image) tuple
        """
        # Decode latent to image
        out = self.vqgan.decode(self.z)

        # Generate cutouts
        cutouts = self.make_cutouts(out)

        # Encode cutouts with CLIP
        image_features = self.clip.encode_image(cutouts)

        # Calculate CLIP loss (normalization happens inside)
        loss = calculate_clip_loss(image_features, self.text_features)

        # Add init image loss if applicable
        if self.z_init is not None and self.init_weight > 0:
            init_loss = F.mse_loss(self.z, self.z_init) * self.init_weight
            loss = loss + init_loss

        return loss, out

    def decode_current(self) -> Image.Image:
        """
        Decode current latent to PIL Image.

        Returns:
            PIL Image
        """
        with torch.no_grad():
            out = self.vqgan.decode(self.z)
            out = out[0].cpu().clamp(-1, 1)
            out = (out + 1) / 2  # [-1, 1] to [0, 1]
            out = TF.to_pil_image(out)
        return out


# ============================================================================
# Utility Functions
# ============================================================================

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    tensor = tensor.cpu().clamp(-1, 1)
    tensor = (tensor + 1) / 2
    return TF.to_pil_image(tensor[0])


def save_image(image: Image.Image, path: str):
    """Save PIL Image to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Saved: {path}")
