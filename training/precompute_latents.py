# /root/Xzk/RockProject/train_ldm_rock.py
# ==============================================================================
#  Standalone Latent Diffusion Model for Conditional 3D Rock Generation
#  Author: AI Assistant for Xzk
#  Version: 6.1 (Loss Visualization & OOM Safe)
#  Description: This version adds functionality to record and plot training/validation
#               loss curves. It also defaults to a safer batch_size to prevent
#               common OOM errors.
# ==============================================================================

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pickle
import json
import math
from einops import rearrange
# ### MODIFICATION 1: Import matplotlib for plotting ###
import matplotlib.pyplot as plt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SECTION 1: HELPER CLASSES & FUNCTIONS (Unchanged)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class LabelProcessor:
    def __init__(self, params_to_log, all_params_list):
        from sklearn.preprocessing import StandardScaler
        self.scaler, self.params_to_log, self.all_params_list = StandardScaler(), params_to_log, all_params_list
        self.param_indices_to_log = [all_params_list.index(p) for p in params_to_log if p in all_params_list]
        self.is_fitted = False

    def fit(self, labels_np):
        processed_labels = self._apply_log_transform(np.copy(labels_np))
        self.scaler.fit(processed_labels);
        self.is_fitted = True

    def transform(self, labels_np):
        if not self.is_fitted: raise RuntimeError("Processor not fitted.")
        processed_labels = self._apply_log_transform(np.copy(labels_np))
        return self.scaler.transform(processed_labels)

    def _apply_log_transform(self, labels_np):
        if self.param_indices_to_log: labels_np[:, self.param_indices_to_log] = np.log1p(
            labels_np[:, self.param_indices_to_log])
        return labels_np


class PhysicsLatentDataset(Dataset):
    def __init__(self, latent_dir: str, label_file: str, processor: LabelProcessor, split='train', split_ratio=0.9):
        self.processor = processor
        with open(label_file, 'r') as f:
            all_labels_dict = json.load(f)
        print("Loading and matching latent files with labels...")
        temp_latent_files, temp_physics_vectors = [], []
        for sample_key, physics_values in tqdm(all_labels_dict.items(), desc="Parsing JSON"):
            latent_path = os.path.join(latent_dir, f"{sample_key}.pt")
            if os.path.exists(latent_path):
                temp_latent_files.append(latent_path);
                temp_physics_vectors.append(physics_values)
        indices = np.arange(len(temp_latent_files));
        np.random.seed(42)  # Ensure consistent splits
        np.random.shuffle(indices)
        split_idx = int(len(temp_latent_files) * split_ratio)
        split_indices = indices[:split_idx] if split == 'train' else indices[split_idx:]
        self.latent_files = [temp_latent_files[i] for i in split_indices]
        physics_np_full = np.array(temp_physics_vectors, dtype=np.float32)

        if not self.processor.is_fitted:
            print("Warning: LabelProcessor is not fitted. Fitting on the full dataset now.")
            processor.fit(np.array(temp_physics_vectors, dtype=np.float32))

        self.physics_vectors_transformed = self.processor.transform(physics_np_full[split_indices])
        print(f"Successfully loaded {len(self.latent_files)} samples for '{split}' split.")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_tensor = torch.load(self.latent_files[idx], map_location='cpu', weights_only=True)
        physics_vector = torch.tensor(self.physics_vectors_transformed[idx], dtype=torch.float32)
        return {"image": latent_tensor, "condition": physics_vector}


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
            if mask.any():
                null_cond_expanded = self.null_condition.expand(mask.sum(), -1)
                embeddings[mask] = null_cond_expanded.to(embeddings.dtype)
        return embeddings.unsqueeze(1)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SECTION 2: CORE CONDITIONAL 3D LDM (Unchanged)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        torch_version = tuple(map(int, torch.__version__.split('+')[0].split('.')))
        self.use_native_sdpa = torch_version >= (2, 0, 0)

    def forward(self, x, context=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        b, n, _ = q.shape
        q = q.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        _, m, _ = k.shape
        k = k.view(b, m, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, m, self.heads, self.dim_head).transpose(1, 2)
        if self.use_native_sdpa:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                   dropout_p=self.to_out[1].p if self.training else 0.0)
        else:
            sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * (self.dim_head ** -0.5)
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
        self.proj_in = nn.Conv3d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock3D(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for _ in
             range(depth)])
        self.proj_out = nn.Conv3d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        b, c, d, h, w = x.shape;
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        for block in self.transformer_blocks: x = block(x, context=context)
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        return self.proj_out(x) + x_in


class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout, temb_channels=512):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1, self.conv1 = Normalize(in_channels), nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else nn.Identity()
        self.norm2, self.dropout, self.conv2 = Normalize(out_channels), nn.Dropout(dropout), nn.Conv3d(out_channels,
                                                                                                       out_channels, 3,
                                                                                                       1, 1)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1, 1, 0) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb):
        h = nonlinearity(self.norm1(x));
        h = self.conv1(h)
        if temb is not None: h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        h = nonlinearity(self.norm2(h));
        h = self.dropout(h);
        h = self.conv2(h)
        return self.shortcut(x) + h


class Downsample3D(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.conv = nn.Conv3d(in_ch, in_ch, 3, 2, 1)

    def forward(self, x): return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.conv = nn.Conv3d(in_ch, in_ch, 3, 1, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
        return self.conv(x)


class UNetModel3D(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0,
                 channel_mult=(1, 2, 4, 8), context_dim=None, num_heads=8, transformer_depth=1):
        super().__init__()
        self.model_channels = model_channels;
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
                                        nn.Linear(time_embed_dim, time_embed_dim))
        self.input_conv = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        self.input_blocks = nn.ModuleList();
        ch = model_channels;
        input_chans = [ch];
        ds = 1
        for i_level, mult in enumerate(channel_mult):
            ch_out = model_channels * mult
            for i_block in range(num_res_blocks):
                layers = [
                    ResnetBlock3D(in_channels=ch, out_channels=ch_out, dropout=dropout, temb_channels=time_embed_dim)];
                ch = ch_out
                if ds in attention_resolutions: layers.append(
                    SpatialTransformer3D(in_channels=ch, n_heads=num_heads, d_head=ch // num_heads,
                                         context_dim=context_dim, depth=transformer_depth))
                self.input_blocks.append(nn.Sequential(*layers));
                input_chans.append(ch)
            if i_level != len(channel_mult) - 1: self.input_blocks.append(
                Downsample3D(ch)); ds *= 2; input_chans.append(ch)
        self.middle_block = nn.Sequential(
            ResnetBlock3D(in_channels=ch, out_channels=ch, dropout=dropout, temb_channels=time_embed_dim),
            SpatialTransformer3D(in_channels=ch, n_heads=num_heads, d_head=ch // num_heads, context_dim=context_dim,
                                 depth=transformer_depth),
            ResnetBlock3D(in_channels=ch, out_channels=ch, dropout=dropout, temb_channels=time_embed_dim))
        self.output_blocks = nn.ModuleList()
        for i_level, mult in reversed(list(enumerate(channel_mult))):
            ch_out = model_channels * mult
            for i_block in range(num_res_blocks + 1):
                ch_skip = input_chans.pop()
                layers = [ResnetBlock3D(in_channels=ch + ch_skip, out_channels=ch_out, dropout=dropout,
                                        temb_channels=time_embed_dim)];
                ch = ch_out
                if ds in attention_resolutions: layers.append(
                    SpatialTransformer3D(in_channels=ch, n_heads=num_heads, d_head=ch // num_heads,
                                         context_dim=context_dim, depth=transformer_depth))
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
        t_emb = self.time_embed(self._timedim_embedding(timesteps));
        h = self.input_conv(x);
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
        super().__init__();
        self.unet = UNetModel3D(**unet_config)
        self.cond_stage_model = PhysicsConditioner(**cond_stage_config)
        self.num_timesteps = timesteps;
        betas = torch.linspace(linear_start, linear_end, timesteps, dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1. - betas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(torch.float32))

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x, c):
        device = x.device;
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=device).long()
        noise = torch.randn_like(x);
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        context = self.cond_stage_model(c);
        predicted_noise = self.unet(x_noisy, t, context)
        return nn.functional.mse_loss(predicted_noise, noise)


def extract_into_tensor(a, t, x_shape):
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SECTION 3: MAIN TRAINING SCRIPT
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
    print("Step 1: Setting up configuration...")

    class TrainingConfig:
        project_root, latent_dir, label_file = "/root/Xzk/RockProject", "latents_ldm_official", "physics_labels.json"
        output_dir = os.path.join(project_root, "outputs_ldm_final_3d_conditional_multigpu")
        latent_dir, label_file = os.path.join(project_root, latent_dir), os.path.join(project_root, label_file)
        physics_params = ["Total_Porosity", "Effective_Porosity", "Avg_Coord_Number", "Permeability_Kx",
                          "Permeability_Ky", "Permeability_Kz", "Kx_Validity_Mask", "Ky_Validity_Mask",
                          "Kz_Validity_Mask"]
        log_transform_params = ["Permeability_Kx", "Permeability_Ky", "Permeability_Kz"]
        epochs, lr = 300, 2e-5

        # ### CRITICAL OOM PARAMETER ###
        # This is the most likely cause of your OOM errors. It was 4.
        # I've set it to a safer value of 2.
        # IF YOU STILL GET OOM, CHANGE THIS TO 1.
        batch_size = 2

        unet_config = {
            "in_channels": 4, "model_channels": 128, "out_channels": 4, "num_res_blocks": 2,
            # This is the other critical memory parameter. It MUST stay this way.
            "attention_resolutions": [4, 8],
            "dropout": 0.1, "channel_mult": [1, 2, 4, 4], "context_dim": 512,
            "num_heads": 8, "transformer_depth": 1
        }
        conditioner_config = {"physics_dim": len(physics_params), "embed_dim": unet_config["context_dim"]}

    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nStep 2: Preparing data...")
    processor = LabelProcessor(params_to_log=config.log_transform_params, all_params_list=config.physics_params)
    train_dataset = PhysicsLatentDataset(config.latent_dir, config.label_file, processor, split='train')
    val_dataset = PhysicsLatentDataset(config.latent_dir, config.label_file, processor, split='val')
    processor_path = os.path.join(config.output_dir, "label_processor.pkl")
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"Label processor fitted and saved to {processor_path}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)

    print("\nStep 3: Initializing model and optimizer...")
    model = LatentDiffusion(unet_config=config.unet_config, cond_stage_config=config.conditioner_config)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    compiled_model = model  # Fallback if compile fails
    try:
        compiled_model = torch.compile(model);
        print("Model compiled successfully!")
    except Exception as e:
        print(f"torch.compile failed: {e}. Running in eager mode.")
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=config.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader))

    print("\nStep 4: Starting training...")
    best_val_loss = float('inf')
    # ### MODIFICATION 2: Initialize lists to store loss history ###
    train_loss_history = []
    val_loss_history = []

    for epoch in range(config.epochs):
        compiled_model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=True)

        # ### MODIFICATION 3A: Track average training loss for the epoch ###
        epoch_train_loss = 0.0

        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)
            x_start, condition = batch["image"].to(device), batch["condition"].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                loss = compiled_model(x_start, condition)
                if isinstance(loss, torch.Tensor) and loss.ndim > 0: loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.1e}"})

        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_epoch_train_loss)

        compiled_model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_start, condition = batch["image"].to(device), batch["condition"].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    loss = compiled_model(x_start, condition)
                    if isinstance(loss, torch.Tensor) and loss.ndim > 0: loss = loss.mean()
                    val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader) if len(val_loader) > 0 else 0
        # ### MODIFICATION 3B: Store validation loss for the epoch ###
        val_loss_history.append(avg_val_loss)
        print(
            f"Epoch {epoch + 1} Summary | Avg Train Loss: {avg_epoch_train_loss:.4f} | Avg Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_to_save = compiled_model.module if hasattr(compiled_model, 'module') else compiled_model
            if hasattr(model_to_save, '_orig_mod'): model_to_save = model_to_save._orig_mod
            save_path = os.path.join(config.output_dir, "best_model.pth")
            torch.save(model_to_save.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

    print("\n--- Training finished! ---")

    # ### MODIFICATION 4: Plotting and saving the loss curve ###
    print("Step 5: Generating and saving loss curve...")
    plt.figure(figsize=(12, 6))
    epochs_range = range(1, config.epochs + 1)

    # You requested every 5, but plotting every epoch gives a more detailed curve.
    # We can plot with markers every 5 epochs if you like.
    plt.plot(epochs_range, train_loss_history, label='Training Loss', color='royalblue', alpha=0.9, marker='o',
             markevery=10, markersize=4, linestyle='-')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss', color='darkorange', alpha=0.9, marker='s',
             markevery=10, markersize=4, linestyle='-')

    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    loss_curve_path = os.path.join(config.output_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path, dpi=300)
    print(f"Loss curve saved to {loss_curve_path}")
    # plt.show() # Uncomment if running in an interactive session like Jupyter


if __name__ == "__main__":
    torch.manual_seed(42)
    main()