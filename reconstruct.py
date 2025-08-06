# ==============================================================================
# reconstruct.py
#
# Changelog:
# - v1.3: [Polish] Converted all console output to English, removed redundant logging,
#           and fixed FutureWarnings for torch.load and autocast.
# - v1.2: [Feature] Implemented automated batch processing of the input directory.
# - v1.1: Fixed a SyntaxError.
# - v1.0: Initial merge of reconstruction and npy-to-image conversion logic.
# ==============================================================================

# --- 0. Import Libraries (Unchanged) ---
import os
import argparse
import glob
import numpy as np
import pickle
import json
import math
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchvision import transforms
from einops import rearrange
import time
from contextlib import redirect_stdout
import shutil

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("=" * 80 + "\n[ERROR] scikit-learn is not installed. Please run: pip install scikit-learn\n" + "=" * 80)
    exit()


# All model and function definitions from SECTION 1, 2, and 3 are unchanged
# and are omitted here for brevity. They are included in the final script.

# ==============================================================================
# SECTION 1: Model Definitions (Unchanged)
# ==============================================================================
class LabelProcessor:
    def __init__(self, params_to_log, all_params_list):
        self.scaler, self.params_to_log, self.all_params_list = StandardScaler(), params_to_log, all_params_list
        self.param_indices_to_log = [all_params_list.index(p) for p in params_to_log if p in all_params_list]
        self.is_fitted = False

    def fit(self, labels_np):
        processed_labels = self._apply_log_transform(np.copy(labels_np))
        self.scaler.fit(processed_labels)
        self.is_fitted = True

    def transform(self, labels_np):
        if not self.is_fitted: raise RuntimeError("Processor not fitted.")
        processed_labels = self._apply_log_transform(np.copy(labels_np))
        return self.scaler.transform(processed_labels)

    def _apply_log_transform(self, labels_np):
        if self.param_indices_to_log: labels_np[:, self.param_indices_to_log] = np.log1p(
            labels_np[:, self.param_indices_to_log])
        return labels_np


class PhysicsConditioner(nn.Module):
    def __init__(self, physics_dim: int, embed_dim: int, hidden_dim: int = 512, cfg_dropout_prob: float = 0.1):
        super().__init__()
        self.cfg_dropout_prob = cfg_dropout_prob
        self.mlp = nn.Sequential(nn.Linear(physics_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, embed_dim))
        self.null_condition = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, physics_vectors: torch.Tensor):
        embeddings = self.mlp(physics_vectors)
        if self.training and self.cfg_dropout_prob > 0:
            mask = torch.rand(embeddings.size(0), device=embeddings.device) < self.cfg_dropout_prob
            if mask.any(): embeddings[mask] = self.null_condition.expand(mask.sum(), -1).to(embeddings.dtype)
        return embeddings.unsqueeze(1)


def nonlinearity(x): return x * torch.sigmoid(x)


def Normalize(in_channels): return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention3D(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.heads, self.dim_head = heads, dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.scale = dim_head ** -0.5

    def forward(self, x, context=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        b, n, _ = q.shape
        q = q.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        _, m, _ = k.shape
        k = k.view(b, m, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, m, self.heads, self.dim_head).transpose(1, 2)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class BasicTransformerBlock3D(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention3D(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.attn2 = CrossAttention3D(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head,
                                      dropout=dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer3D(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv3d(in_channels, inner_dim, 1, 1, 0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock3D(inner_dim, n_heads, d_head, dropout, context_dim) for _ in range(depth)])
        self.proj_out = nn.Conv3d(inner_dim, in_channels, 1, 1, 0)

    def forward(self, x, context=None):
        b, c, d, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        for block in self.transformer_blocks: x = block(x, context=context)
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        return self.proj_out(x) + x_in


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, temb_channels=512):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1, self.conv1 = Normalize(in_channels), nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else nn.Identity()
        self.norm2, self.dropout, self.conv2 = Normalize(out_channels), nn.Dropout(dropout), nn.Conv3d(out_channels,
                                                                                                       out_channels, 3,
                                                                                                       1, 1)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1, 1, 0) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb):
        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)
        if temb is not None: h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        h = nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class Downsample3D(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.conv = nn.Conv3d(in_ch, in_ch, 3, 2, 1)

    def forward(self, x): return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.conv = nn.Conv3d(in_ch, in_ch, 3, 1, 1)

    def forward(self, x): x = F.interpolate(x, scale_factor=2.0, mode="trilinear",
                                            align_corners=False); return self.conv(x)


class UNetModel3D(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0,
                 channel_mult=(1, 2, 4, 8), context_dim=None, num_heads=8, transformer_depth=1):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
                                        nn.Linear(time_embed_dim, time_embed_dim))
        self.input_conv = nn.Conv3d(in_channels, model_channels, 3, padding=1)
        self.input_blocks = nn.ModuleList()
        ch = model_channels
        input_chans = [ch]
        ds = 1
        for i_level, mult in enumerate(channel_mult):
            ch_out = model_channels * mult
            for i_block in range(num_res_blocks):
                layers = [
                    ResnetBlock3D(in_channels=ch, out_channels=ch_out, dropout=dropout, temb_channels=time_embed_dim)]
                ch = ch_out
                if ds in attention_resolutions: d_head = ch // num_heads; layers.append(
                    SpatialTransformer3D(ch, num_heads, d_head, transformer_depth, dropout, context_dim))
                self.input_blocks.append(nn.Sequential(*layers))
                input_chans.append(ch)
            if i_level != len(channel_mult) - 1: self.input_blocks.append(
                Downsample3D(ch)); ds *= 2; input_chans.append(ch)
        d_head = ch // num_heads
        self.middle_block = nn.Sequential(ResnetBlock3D(ch, ch, dropout, time_embed_dim),
                                          SpatialTransformer3D(ch, num_heads, d_head, transformer_depth, dropout,
                                                               context_dim),
                                          ResnetBlock3D(ch, ch, dropout, time_embed_dim))
        self.output_blocks = nn.ModuleList()
        for i_level, mult in reversed(list(enumerate(channel_mult))):
            ch_out = model_channels * mult
            for i_block in range(num_res_blocks + 1):
                ch_skip = input_chans.pop()
                layers = [ResnetBlock3D(ch + ch_skip, ch_out, dropout, time_embed_dim)]
                ch = ch_out
                if ds in attention_resolutions: d_head = ch // num_heads; layers.append(
                    SpatialTransformer3D(ch, num_heads, d_head, transformer_depth, dropout, context_dim))
                self.output_blocks.append(nn.Sequential(*layers))
            if i_level != 0: self.output_blocks.append(Upsample3D(ch)); ds //= 2
        self.out = nn.Sequential(Normalize(ch), nn.SiLU(), nn.Conv3d(ch, out_channels, 3, 1, 1))

    def _timedim_embedding(self, timesteps):
        half = self.model_channels // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(end=half, dtype=torch.float32) / half).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1) if self.model_channels % 2 == 0 else torch.cat(
            [torch.cos(args), torch.sin(args), torch.zeros_like(args[:, :1])], dim=-1)

    def forward(self, x, timesteps, context=None):
        t_emb = self.time_embed(self._timedim_embedding(timesteps))
        h = self.input_conv(x)
        hs = [h]
        for module in self.input_blocks:
            if isinstance(module, Downsample3D):
                h = module(h)
            else:
                for layer in module: h = layer(h, t_emb) if isinstance(layer, ResnetBlock3D) else layer(h, context)
            hs.append(h)
        for layer in self.middle_block: h = layer(h, t_emb) if isinstance(layer, ResnetBlock3D) else layer(h, context)
        for module in self.output_blocks:
            if isinstance(module, Upsample3D):
                h = module(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in module: h = layer(h, t_emb) if isinstance(layer, ResnetBlock3D) else layer(h, context)
        return self.out(h)


class LatentDiffusion(nn.Module):
    def __init__(self, unet_config, cond_stage_config, timesteps=1000, linear_start=0.00085, linear_end=0.0120):
        super().__init__()
        self.unet = UNetModel3D(**unet_config)
        self.cond_stage_model = PhysicsConditioner(**cond_stage_config)
        self.num_timesteps = timesteps
        betas = torch.linspace(linear_start, linear_end, timesteps, dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1. - betas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(torch.float32))


# ==============================================================================
# SECTION 2: Core Functions (Refined Output)
# ==============================================================================
def calculate_physics_vector(image_folder_path: str) -> np.ndarray:
    print("Step 1/5: Calculating Physical Properties...")
    start_time = time.time()
    try:
        import porespy as ps;
        import openpnm as op
    except ImportError:
        print("[ERROR] Porespy or OpenPNM not installed. Please run: pip install porespy openpnm")
        exit()
    image_paths = sorted(glob.glob(os.path.join(image_folder_path, '*.png')))
    if not image_paths: raise FileNotFoundError(f"No PNG files found in '{image_folder_path}'.")

    print("  - Loading and binarizing images...")
    volume_3d = np.stack([np.array(Image.open(path).convert("L")) for path in image_paths], axis=0)
    volume_3d_np = (volume_3d < np.mean(volume_3d))

    total_porosity = ps.metrics.porosity(volume_3d_np)
    if total_porosity >= 1.0 or total_porosity <= 0.0:
        props = {"Total_Porosity": total_porosity, "Effective_Porosity": total_porosity, "Avg_Coord_Number": 0.0,
                 "Permeability_Kx": 0.0, "Permeability_Ky": 0.0, "Permeability_Kz": 0.0,
                 "Kx_Validity_Mask": 0.0, "Ky_Validity_Mask": 0.0, "Kz_Validity_Mask": 0.0}
    else:
        eff_im = np.logical_not(ps.filters.find_disconnected_voxels(im=volume_3d_np, conn=26))
        effective_porosity = ps.metrics.porosity(eff_im)
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            snow = ps.networks.snow2(eff_im);
            pn = op.io.network_from_porespy(snow.network)
        avg_coord = np.mean(pn.num_neighbors(pn.pores())) if pn.Np > 0 else 0.0
        props = {"Total_Porosity": total_porosity, "Effective_Porosity": effective_porosity,
                 "Avg_Coord_Number": avg_coord}
        for i, axis_name in enumerate(['x', 'y', 'z']):
            try:
                tort = ps.simulations.tortuosity_fd(im=eff_im, axis=i).tortuosity
            except Exception:
                tort = np.inf
            K_md = (effective_porosity * (1e-6 ** 2)) / tort * 1e15 if tort > 0 and not np.isinf(tort) else 0.0
            props[f'Permeability_K{axis_name}'] = K_md
            props[f'K{axis_name}_Validity_Mask'] = 1.0 if K_md > 0 else 0.0

    ordered_keys = ["Total_Porosity", "Effective_Porosity", "Avg_Coord_Number", "Permeability_Kx", "Permeability_Ky",
                    "Permeability_Kz", "Kx_Validity_Mask", "Ky_Validity_Mask", "Kz_Validity_Mask"]
    final_vector = np.array([props[k] for k in ordered_keys], dtype=np.float32).reshape(1, -1)
    print(f"  - Calculation complete. Time: {time.time() - start_time:.2f}s.")
    return final_vector


def encode_rock_to_latent(image_folder_path: str, vae: AutoencoderKL, device: torch.device) -> torch.Tensor:
    print("Step 2/5: Encoding Input Rock to Latent Space...")
    start_time = time.time()
    transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    image_paths = sorted(glob.glob(os.path.join(image_folder_path, '*.png')))
    slices = torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths]).to(device, vae.dtype)
    latents_list = []
    for i in tqdm(range(0, len(slices), 16), desc="  - VAE Encoding"):
        batch = slices[i:i + 16]
        latents_list.append(vae.encode(batch).latent_dist.mean * vae.config.scaling_factor)
    full_latent = torch.cat(latents_list).permute(1, 0, 2, 3).unsqueeze(0)
    latent_32 = F.interpolate(full_latent, size=(32, 32, 32), mode='trilinear', align_corners=False)
    print(f"  - Encoding complete. Latent shape: {latent_32.shape}. Time: {time.time() - start_time:.2f}s")
    return latent_32


def transform_latent_with_ldm(input_latent: torch.Tensor, raw_physics_vector: np.ndarray, ldm_model: LatentDiffusion,
                              processor: LabelProcessor, device: torch.device) -> torch.Tensor:
    print("Step 3/5: Transforming Latent via Diffusion Model...")
    start_time = time.time()
    print("  - Conditioning model with calculated physical properties...")
    processed_physics = processor.transform(raw_physics_vector)
    condition_tensor = torch.tensor(processed_physics, dtype=torch.float32).to(device)
    timesteps = torch.zeros((1,), device=device).long()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == 'cuda')):
        context = ldm_model.cond_stage_model(condition_tensor)
        transformed_latent = ldm_model.unet(input_latent, timesteps, context)
    print(f"  - Transformation complete. Time: {time.time() - start_time:.2f}s")
    return transformed_latent.to(torch.float32)


def decode_latent_to_rock(transformed_latent: torch.Tensor, vae: AutoencoderKL, device: torch.device,
                          recon_size: int = 128) -> np.ndarray:
    print(f"Step 4/5: Decoding Latent to {recon_size}³ Rock Volume...")
    start_time = time.time()
    upscaled = F.interpolate(transformed_latent, size=(recon_size, 64, 64), mode='trilinear',
                             align_corners=False) / vae.config.scaling_factor
    final_slices = []
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == 'cuda')):
        for d in tqdm(range(upscaled.shape[2]), desc=f"  - VAE Decoding ({recon_size} slices)"):
            slice_latent = upscaled[:, :, d, :, :]
            img_slice = vae.decode(slice_latent).sample[0]
            img_slice_norm = (img_slice.clamp(-1, 1) + 1) / 2
            final_slice_gray = img_slice_norm.mean(dim=0, keepdim=True)
            final_slice_resized = F.interpolate(final_slice_gray.unsqueeze(0), size=(recon_size, recon_size),
                                                mode='bilinear', align_corners=False).squeeze()
            final_slices.append(final_slice_resized.cpu().to(torch.float32))
    final_rock_numpy = torch.stack(final_slices, dim=0).numpy()
    print(f"  - Decoding complete. Final volume shape: {final_rock_numpy.shape}. Time: {time.time() - start_time:.2f}s")
    return final_rock_numpy


# ==============================================================================
# SECTION 3: NPY to Image Conversion (Refined Output)
# ==============================================================================
def npy_to_images(npy_path, output_dir):
    print(f"\nConverting .npy to images...")

    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:  # Clear existing content
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # Load .npy file
    try:
        rock_volume = np.load(npy_path)
    except Exception as e:
        print(f"[ERROR] Failed to load .npy file: {e}")
        return

    # Ensure 3D volume
    if rock_volume.ndim == 4:
        if rock_volume.shape[1] > 1:
            rock_volume = np.mean(rock_volume, axis=1)
        else:
            rock_volume = np.squeeze(rock_volume, axis=1)
    if rock_volume.ndim != 3:
        print(f"[ERROR] Volume is not 3D (shape: {rock_volume.shape}). Cannot convert to images.")
        return

    # Binarize and save slices
    threshold = np.mean(rock_volume)
    num_slices = rock_volume.shape[0]
    print(f"  - Saving {num_slices} slices using threshold {threshold:.4f}...")
    for i in tqdm(range(num_slices), desc="  - Converting to PNG"):
        slice_2d = rock_volume[i, :, :]
        binary_slice = (slice_2d > threshold).astype(np.uint8) * 255
        img = Image.fromarray(binary_slice, 'L')
        output_filename = os.path.join(output_dir, f"slice_{i:04d}.png")
        img.save(output_filename)

    print(f"  - ✅ Conversion complete. {num_slices} images saved to '{output_dir}'")


# ==============================================================================
# SECTION 4: Main Execution Block (v1.3)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PILO: Physics-Informed 3D Rock Reconstruction (64³ -> 128³)")
    parser.add_argument('--input', type=str, default="./input/",
                        help="Path to the root directory containing one or more rock sample folders.")
    args = parser.parse_args()

    # --- Hardcoded Paths & Parameters ---
    OUTPUT_DIR = "./output/"
    LDM_MODEL_PATH = "./models/ldm/best_model.pth"
    PROCESSOR_PATH = "./models/ldm/label_processor.pkl"
    VAE_MODEL_PATH = "./models/vae/"
    RECON_SIZE = 128

    # --- Initial Setup ---
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"======================================================")
    print(f"    PILO Rock Reconstruction  (64³ -> {RECON_SIZE}³)")
    print(f"======================================================")
    print(f"Using device: {device}")

    os.makedirs(args.input, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Models (Once) ---
    print("\n--- Loading models and processors... ---")
    for path in [LDM_MODEL_PATH, PROCESSOR_PATH, VAE_MODEL_PATH]:
        if not os.path.exists(path):
            print(f"\n[FATAL ERROR] Critical file/folder not found: '{path}'")
            print("Please ensure you have downloaded and placed all model files correctly as per README.md.")
            exit()

    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH, torch_dtype=torch.float16).to(device).eval()
    with open(PROCESSOR_PATH, 'rb') as f:
        processor = pickle.load(f)

    unet_config = {
        "in_channels": 4, "model_channels": 128, "out_channels": 4, "num_res_blocks": 2,
        "attention_resolutions": [4, 8], "dropout": 0.1, "channel_mult": [1, 2, 4, 4],
        "context_dim": 512, "num_heads": 8, "transformer_depth": 1
    }
    conditioner_config = {"physics_dim": 9, "embed_dim": unet_config["context_dim"]}
    ldm_model = LatentDiffusion(unet_config=unet_config, cond_stage_config=conditioner_config)

    state_dict = torch.load(LDM_MODEL_PATH, map_location='cpu', weights_only=True)  # Set weights_only=True
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    ldm_model.load_state_dict(new_state_dict, strict=True)
    ldm_model.to(device).eval()
    print("--- Models loaded successfully. Ready to process samples. ---\n")

    # --- Scan and Process Input Directory ---
    input_root = args.input
    sample_dirs = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])

    if not sample_dirs:
        print(f"[WARNING] Input directory '{input_root}' is empty. Please add sample folders to process.")
        return

    for sample_name in sample_dirs:
        sample_folder_path = os.path.join(input_root, sample_name)

        print(f"======================================================")
        print(f"Processing sample: {sample_name}")

        # Validate sample folder
        image_files = glob.glob(os.path.join(sample_folder_path, '*.png'))
        if len(image_files) != 64:
            print(
                f"[SKIPPING] Sample '{sample_name}' skipped. Reason: 64 PNG files required, found {len(image_files)}.")
            continue

        # --- Execute Reconstruction Workflow ---
        with torch.no_grad():
            raw_physics_vector = calculate_physics_vector(sample_folder_path)
            input_latent = encode_rock_to_latent(sample_folder_path, vae, device)
            transformed_latent = transform_latent_with_ldm(input_latent, raw_physics_vector, ldm_model, processor,
                                                           device)
            final_rock_volume = decode_latent_to_rock(transformed_latent, vae, device, RECON_SIZE)

        # --- Save Results ---
        print("\nStep 5/5: Saving Final Results...")
        output_npy_path = os.path.join(OUTPUT_DIR, f"out_{sample_name}.npy")
        np.save(output_npy_path, final_rock_volume)

        image_output_dir = os.path.join(OUTPUT_DIR, f"out_{sample_name}_images")
        npy_to_images(output_npy_path, image_output_dir)
        print(f"--- ✅ Sample '{sample_name}' processed successfully ---")

    print(f"\n\n======================================================")
    print(f"🎉 All valid samples have been processed!")
    print(f"   - All outputs saved in: {OUTPUT_DIR}")
    print(f"   - Total elapsed time: {time.time() - total_start_time:.2f} seconds")
    print(f"======================================================")


if __name__ == '__main__':
    main()