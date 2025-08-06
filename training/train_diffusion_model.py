# --------------------------------------------------------------------------
# precompute_latents.py (v2)
#
# 目的: 使用预训练的自研VQ-VAE模型，将原始的64^3岩心数据批量编码为
#      16^3的潜在空间特征，并保存为新的数据集。
#
# 作者: RockProject Team
# 日期: 2023-10-27
# 更新(v2): 修正了预训练模型权重的正确路径。
# --------------------------------------------------------------------------

# ==============================================================================
# 0. IMPORTS (必要的库)
# ==============================================================================
import os
import argparse
import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.io import imread


# ==============================================================================
# 1. MODEL DEFINITIONS (从 train_final.py 精确复制)
#    这部分代码确保了我们加载模型时所用的架构，与训练时完全一致。
# ==============================================================================

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, C, D, H, W = x.size();
        p_q = self.query_conv(x).view(b, -1, D * H * W).permute(0, 2, 1);
        p_k = self.key_conv(x).view(b, -1, D * H * W)
        energy = torch.bmm(p_q, p_k);
        attention = self.softmax(energy);
        p_v = self.value_conv(x).view(b, -1, D * H * W)
        out = torch.bmm(p_v, attention.permute(0, 2, 1));
        out = out.view(b, C, D, H, W);
        out = self.gamma * out + x;
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                                          nn.BatchNorm3d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__();
        self._e_dim = embedding_dim;
        self._n_emb = num_embeddings;
        self._c_cost = commitment_cost
        self._embedding = nn.Embedding(self._n_emb, self._e_dim);
        self._embedding.weight.data.uniform_(-1 / self._n_emb, 1 / self._n_emb)

    def forward(self, inputs):
        inputs_p = inputs.permute(0, 2, 3, 4, 1).contiguous();
        input_shape = inputs_p.shape;
        flat_input = inputs_p.view(-1, self._e_dim)
        distances = (torch.sum(flat_input ** 2, 1, True) + torch.sum(self._embedding.weight ** 2, 1) - 2 * torch.matmul(
            flat_input, self._embedding.weight.t()))
        encoding_indices = torch.argmin(distances, 1).unsqueeze(1);
        encodings = torch.zeros(encoding_indices.shape[0], self._n_emb, device=inputs.device).scatter_(1,
                                                                                                       encoding_indices,
                                                                                                       1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape);
        e_loss = F.mse_loss(quantized.detach(), inputs_p);
        q_loss = F.mse_loss(quantized, inputs_p.detach())
        loss = q_loss + self._c_cost * e_loss;
        quantized = inputs_p + (quantized - inputs_p).detach();
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous()


class EncoderV2(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, embedding_dim):
        super().__init__();
        self.init_conv = nn.Conv3d(in_channels, num_hiddens // 2, 4, 2, 1);
        self.res1 = nn.Sequential(*[ResBlock(num_hiddens // 2, num_hiddens // 2) for _ in range(num_residual_layers)])
        self.down1 = nn.Conv3d(num_hiddens // 2, num_hiddens, 4, 2, 1);
        self.res2 = nn.Sequential(*[ResBlock(num_hiddens, num_hiddens) for _ in range(num_residual_layers)])
        self.attn = SelfAttention(num_hiddens);
        self.to_b = nn.Conv3d(num_hiddens, embedding_dim, 1);
        self.to_t = nn.Conv3d(num_hiddens, embedding_dim, 1)

    def forward(self, x):
        x = F.relu(self.init_conv(x))
        x = self.res1(x)
        b = self.down1(x)
        b = self.res2(b)
        a = self.attn(b)
        z_b = self.to_b(a)
        z_t = self.to_t(F.adaptive_avg_pool3d(a, 1))
        return z_b, z_t


class DecoderV2(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers):
        super().__init__();
        self.from_l = nn.Conv3d(in_channels, num_hiddens, 1);
        self.res1 = nn.Sequential(*[ResBlock(num_hiddens, num_hiddens) for _ in range(num_residual_layers)])
        self.up1 = nn.ConvTranspose3d(num_hiddens, num_hiddens // 2, 4, 2, 1);
        self.res2 = nn.Sequential(*[ResBlock(num_hiddens // 2, num_hiddens // 2) for _ in range(num_residual_layers)])
        self.final_conv = nn.ConvTranspose3d(num_hiddens // 2, 1, 4, 2, 1)

    def forward(self, z_b, z_t):
        z_t_up = F.interpolate(z_t, size=z_b.shape[2:], mode='trilinear', align_corners=False)
        z_f = z_b + z_t_up
        x = self.from_l(z_f)
        x = self.res1(x)
        x = F.relu(self.up1(x))
        x = self.res2(x)
        return torch.sigmoid(self.final_conv(x))


class Generator(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.encoder = EncoderV2(in_channels, num_hiddens, num_residual_layers, embedding_dim)
        self.vq_b = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.vq_t = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = DecoderV2(embedding_dim, num_hiddens, num_residual_layers)

    def forward(self, x):
        z_b, z_t = self.encoder(x)
        vq_l_b, q_b = self.vq_b(z_b)
        vq_l_t, q_t = self.vq_t(z_t)
        total_vq_l = vq_l_b + vq_l_t
        x_recon = self.decoder(q_b, q_t)
        return total_vq_l, x_recon


# ==============================================================================
# 2. DATASET DEFINITION (从 train_final.py 精确复制)
# ==============================================================================
class RockVolumeDataset(Dataset):
    def __init__(self, directory):
        self.sample_dirs = sorted(
            [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        print(f"Found {len(self.sample_dirs)} samples in {directory}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        image_paths = sorted(glob.glob(os.path.join(sample_dir, '*.png')))
        if len(image_paths) != 64:
            print(f"Warning: Skipping sample {sample_dir} as it has {len(image_paths)} slices instead of 64.")
            return None
        volume_slices = [imread(p, as_gray=True) for p in image_paths]
        volume_3d = np.stack(volume_slices, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(volume_3d).unsqueeze(0)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ==============================================================================
# 3. MAIN SCRIPT LOGIC
# ==============================================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Generator(
        in_channels=1,
        num_hiddens=args.hidden_dims,
        num_residual_layers=args.residual_layers,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost
    ).to(device)

    # 忽略 FutureWarning
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"VQ-VAE model weights loaded from {args.model_path}")

    dataset = RockVolumeDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn,
                            pin_memory=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    all_latents_bottom = []
    all_latents_top = []

    print("\nStarting to compute latents...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding Batches"):
            if batch is None:
                continue

            batch = batch.to(device)
            z_bottom, z_top = model.encoder(batch)
            all_latents_bottom.append(z_bottom.cpu().numpy())
            all_latents_top.append(z_top.cpu().numpy())

    if not all_latents_bottom:
        print("Error: No valid data was processed. Please check your data directory and format.")
        return

    final_latents_bottom = np.concatenate(all_latents_bottom, axis=0)
    final_latents_top = np.concatenate(all_latents_top, axis=0)

    output_path_bottom = args.output_path.replace('.npy', '_bottom.npy')
    output_path_top = args.output_path.replace('.npy', '_top.npy')

    np.save(output_path_bottom, final_latents_bottom)
    np.save(output_path_top, final_latents_top)

    print("\n--- Computation Complete! ---")
    print(f"Bottom latents (main spatial info) saved to: {output_path_bottom}")
    print(f"Shape: {final_latents_bottom.shape}")
    print(f"Top latents (global info) saved to: {output_path_top}")
    print(f"Shape: {final_latents_top.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute latents using a pre-trained VQ-VAE.")

    parser.add_argument('--data_path', type=str, default='/root/Xzk/RockProject/dataset_500_samples',
                        help="Path to the original 64^3 rock data directory.")

    #
    # ------------------- 唯一的、关键的修改在这里 -------------------
    #
    parser.add_argument('--model_path', type=str,
                        default='/root/Xzk/RockProject/checkpoints_archive_2560_epochs/generator_latest.pth',
                        help="Path to the pre-trained generator .pth file.")
    #
    # -------------------------------------------------------------------
    #

    parser.add_argument('--output_path', type=str,
                        default='/root/Xzk/RockProject/data/latents_from_vae/rock_latents.npy',
                        help="Path to save the computed latent features (.npy format).")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for encoding.")
    parser.add_argument('--hidden_dims', type=int, default=192)
    parser.add_argument('--residual_layers', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_embeddings', type=int, default=1024)
    parser.add_argument('--commitment_cost', type=float, default=0.25)

    args = parser.parse_args()
    main(args)

