import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_spectra_from_excel(
    excel_path: str,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    df = pd.read_excel(excel_path)

    spec_df = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    prop_series = pd.to_numeric(df.iloc[:, -1], errors="coerce")

    spectra = spec_df.values.astype(np.float32)
    prop = prop_series.values.astype(np.float32).reshape(-1, 1)

    mask_valid = (~np.isnan(spectra).any(axis=1)) & (~np.isnan(prop).any(axis=1))
    if mask_valid.sum() < len(spectra):
        print(f"Dropping {len(spectra) - int(mask_valid.sum())} rows with NaNs.")

    spectra = spectra[mask_valid]
    prop = prop[mask_valid]

    spec_min = float(np.nanmin(spectra))
    spec_max = float(np.nanmax(spectra))
    spectra_norm = (spectra - spec_min) / (spec_max - spec_min + 1e-8)
    spectra_tanh = spectra_norm * 2.0 - 1.0

    prop_min = float(np.nanmin(prop))
    prop_max = float(np.nanmax(prop))

    X = torch.from_numpy(spectra_tanh).to(device)
    y = torch.from_numpy(prop).to(device)

    print(f"Loaded spectra from: {excel_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Reflectance (mapped) range: {X.min().item():.3f} to {X.max().item():.3f}")
    print(f"Property range: {y.min().item():.3f} to {y.max().item():.3f}")

    return X, y, spec_min, spec_max, prop_min, prop_max


class JointDataset(Dataset):
    def __init__(self, XY_norm: torch.Tensor):
        self.XY = XY_norm

    def __len__(self) -> int:
        return self.XY.shape[0]

    def __getitem__(self, idx: int):
        return self.XY[idx]


class SelfAttention1D(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "d_model must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, need_attn: bool = False):
        B, L, C = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.view(B, L, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        if need_attn:
            return out, attn
        return out, None


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TransformerBlock1DWithAttn(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttention1D(d_model, heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.last_attn = None

    def forward(self, src: torch.Tensor, need_attn: bool = False) -> torch.Tensor:
        x_norm = self.norm1(src)
        attn_out, attn_weights = self.self_attn(x_norm, need_attn=need_attn)
        src2 = src + self.dropout1(attn_out)

        x = self.norm2(src2)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        out = src2 + self.dropout2(x)

        if need_attn:
            self.last_attn = attn_weights.detach()
        else:
            self.last_attn = None

        return out


class Generator(nn.Module):
    def __init__(self, latent_dim: int, seq_len_total: int, d_model: int = 96, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.seq_len = seq_len_total
        self.d_model = d_model

        self.fc_in = nn.Linear(latent_dim, seq_len_total * d_model)
        self.pos_enc = PositionalEncoding1D(d_model, max_len=seq_len_total)

        self.blocks = nn.ModuleList([
            TransformerBlock1DWithAttn(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor, attn_layer_idx: int = None) -> torch.Tensor:
        B = z.size(0)
        x = self.fc_in(z).view(B, self.seq_len, self.d_model)
        x = self.pos_enc(x)

        for i, blk in enumerate(self.blocks):
            need_attn = (attn_layer_idx is not None and i == attn_layer_idx)
            x = blk(x, need_attn=need_attn)

        x = self.fc_out(x).squeeze(-1)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, seq_len_total: int, d_model: int = 96, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len_total
        self.d_model = d_model

        self.fc_in = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding1D(d_model, max_len=seq_len_total)

        self.blocks = nn.ModuleList([
            TransformerBlock1DWithAttn(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, attn_layer_idx: int = None) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.fc_in(x)
        x = self.pos_enc(x)

        for i, blk in enumerate(self.blocks):
            need_attn = (attn_layer_idx is not None and i == attn_layer_idx)
            x = blk(x, need_attn=need_attn)

        x_mean = x.mean(dim=1)
        score = self.fc_out(x_mean)
        return score


def gradient_penalty(
    D: Discriminator,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: str = "cpu",
    lambda_gp: float = 10.0
) -> torch.Tensor:
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates, device=device)

    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp


def spectral_angle_mapper(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = (x * y).sum(dim=1)
    den = (x.norm(dim=1) * y.norm(dim=1)).clamp_min(eps)
    cos = (num / den).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.arccos(cos)


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, device: str = "cpu") -> float:
    x = x.to(device)
    y = y.to(device)

    z = torch.cat([x, y], dim=0)
    dists = torch.cdist(z, z, p=2) ** 2
    mask = ~torch.eye(z.size(0), dtype=torch.bool, device=device)
    median_val = dists[mask].median()
    sigma2 = (median_val / 2.0).clamp_min(1e-6)

    K_xx = torch.exp(-torch.cdist(x, x, p=2) ** 2 / (2 * sigma2))
    K_yy = torch.exp(-torch.cdist(y, y, p=2) ** 2 / (2 * sigma2))
    K_xy = torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * sigma2))

    mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd2.item())


def compute_svd_distance(x: torch.Tensor, y: torch.Tensor, top_k: int = 20, device: str = "cpu") -> float:
    x = x.to(device)
    y = y.to(device)

    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)

    cov_x = x_c.t().mm(x_c) / (x_c.size(0) - 1)
    cov_y = y_c.t().mm(y_c) / (y_c.size(0) - 1)

    s_x = torch.linalg.svdvals(cov_x)
    s_y = torch.linalg.svdvals(cov_y)

    k = min(top_k, s_x.size(0), s_y.size(0))
    dist = torch.norm(s_x[:k] - s_y[:k], p=2)
    return float(dist.item())


def train_wgan_transformer(
    excel_path: str,
    device: str = None,
    latent_dim: int = 64,
    batch_size: int = 16,
    epochs: int = 300,
    lr: float = 1e-4,
    n_critic: int = 5,
    out_dir: str = "wgan_transformer_output"
):
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_spec, y_prop, spec_min, spec_max, prop_min, prop_max = load_spectra_from_excel(
        excel_path, device=device
    )
    seq_len_spec = X_spec.shape[1]
    seq_len_total = seq_len_spec + 1

    y_norm = (y_prop - prop_min) / (prop_max - prop_min + 1e-8) * 2.0 - 1.0
    XY_norm = torch.cat([X_spec, y_norm], dim=1)

    dataset = JointDataset(XY_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(latent_dim=latent_dim, seq_len_total=seq_len_total, d_model=96, nhead=4, num_layers=3).to(device)
    D = Discriminator(seq_len_total=seq_len_total, d_model=96, nhead=4, num_layers=2).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    d_loss_history, g_loss_history = [], []
    mse_history, sam_history = [], []
    mmd_history, svd_history = [], []

    for epoch in range(1, epochs + 1):
        epoch_d_losses, epoch_g_losses = [], []
        for real_xy in dataloader:
            real_xy = real_xy.to(device)

            for _ in range(n_critic):
                z = torch.randn(real_xy.size(0), latent_dim, device=device)
                fake_xy = G(z)
                d_real = D(real_xy)
                d_fake = D(fake_xy.detach())

                gp = gradient_penalty(D, real_xy, fake_xy.detach(), device=device, lambda_gp=10.0)
                d_loss = -(d_real.mean() - d_fake.mean()) + gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            z = torch.randn(real_xy.size(0), latent_dim, device=device)
            gen_xy = G(z)
            d_gen = D(gen_xy)
            g_loss = -d_gen.mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            epoch_d_losses.append(d_loss.item())
            epoch_g_losses.append(g_loss.item())

        d_loss_epoch = float(np.mean(epoch_d_losses))
        g_loss_epoch = float(np.mean(epoch_g_losses))
        d_loss_history.append(d_loss_epoch)
        g_loss_history.append(g_loss_epoch)

        with torch.no_grad():
            eval_n = min(256, X_spec.shape[0])
            idx_real = torch.randperm(X_spec.shape[0], device=device)[:eval_n]
            real_spec_eval = X_spec[idx_real]

            z_eval = torch.randn(eval_n, latent_dim, device=device)
            gen_xy_eval = G(z_eval)
            gen_spec_eval = gen_xy_eval[:, :seq_len_spec]

            real_mean = real_spec_eval.mean(dim=0, keepdim=True)
            gen_mean = gen_spec_eval.mean(dim=0, keepdim=True)

            mse_val = F.mse_loss(gen_mean, real_mean).item()
            sam_val = spectral_angle_mapper(gen_mean, real_mean)[0].item()

            mmd_val = compute_mmd_rbf(real_spec_eval, gen_spec_eval, device=device)
            svd_val = compute_svd_distance(real_spec_eval, gen_spec_eval, top_k=20, device=device)

        mse_history.append(mse_val)
        sam_history.append(sam_val)
        mmd_history.append(mmd_val)
        svd_history.append(svd_val)

        print(
            f"Epoch {epoch:03d} | "
            f"D: {d_loss_epoch:.4f} | G: {g_loss_epoch:.4f} | "
            f"MSE(mean): {mse_val:.3e} | SAM(mean): {sam_val:.3e} | "
            f"MMD: {mmd_val:.3e} | SVD: {svd_val:.3e}"
        )

    epochs_arr = np.arange(1, epochs + 1, dtype=np.int32)
    metrics = np.column_stack([
        epochs_arr,
        d_loss_history,
        g_loss_history,
        mse_history,
        sam_history,
        mmd_history,
        svd_history
    ])
    metrics_path = os.path.join(out_dir, "training_metrics.csv")
    header = "epoch,D_loss,G_loss,MSE_mean,SAM_mean,MMD,SVD_dist"
    np.savetxt(metrics_path, metrics, delimiter=",", header=header, comments="")
    print(f"Saved training metrics to: {metrics_path}")

    def plot_metric(values, ylabel, title, fname):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_arr, values, marker="o", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved figure: {path}")

    plot_metric(mse_history, "MSE between mean real & generated spectra", "MSE(mean) over epochs", "mse_over_epochs.png")
    plot_metric(sam_history, "SAM (radians) between mean real & generated", "SAM(mean) over epochs", "sam_over_epochs.png")
    plot_metric(mmd_history, "MMD (RBF kernel)", "MMD over epochs", "mmd_over_epochs.png")
    plot_metric(svd_history, "SVD-based distance", "SVD distance over epochs", "svd_over_epochs.png")

    return G, D, (X_spec, y_prop, spec_min, spec_max, prop_min, prop_max, seq_len_spec)


def export_attention_and_heatmaps(
    G: Generator,
    latent_dim: int,
    layer_idx: int = 0,
    device: str = "cpu",
    out_dir: str = "wgan_transformer_output/attn_generator"
):
    os.makedirs(out_dir, exist_ok=True)
    G.eval()

    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        _ = G(z, attn_layer_idx=layer_idx)

    attn = G.blocks[layer_idx].last_attn
    if attn is None:
        print("No attention captured; check layer_idx.")
        return

    attn = attn[0].cpu().numpy()
    heads, L, _ = attn.shape

    excel_path = os.path.join(out_dir, f"generator_layer{layer_idx}_attn.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for h in range(heads):
            df = pd.DataFrame(
                attn[h],
                index=[f"q_{i}" for i in range(L)],
                columns=[f"k_{j}" for j in range(L)]
            )
            df.to_excel(writer, sheet_name=f"head{h}", index=True)
    print(f"Saved attention matrices to: {excel_path}")

    for h in range(heads):
        plt.figure(figsize=(5, 4))
        plt.imshow(attn[h], aspect="auto", origin="lower")
        plt.colorbar(label="Attention weight")
        plt.xlabel("Key index (bands + property)")
        plt.ylabel("Query index (bands + property)")
        plt.title(f"Generator attention | layer {layer_idx}, head {h}")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"generator_layer{layer_idx}_head{h}.png")
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Saved attention heatmap: {png_path}")

    attn_mean = attn.mean(axis=0)
    plt.figure(figsize=(5, 4))
    plt.imshow(attn_mean, aspect="auto", origin="lower")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key index (bands + property)")
    plt.ylabel("Query index (bands + property)")
    plt.title(f"Generator attention (mean over {heads} heads) | layer {layer_idx}")
    plt.tight_layout()
    mean_png = os.path.join(out_dir, f"generator_layer{layer_idx}_mean_heads.png")
    plt.savefig(mean_png, dpi=300)
    plt.close()
    print(f"Saved mean-attention heatmap: {mean_png}")


def export_generated_spectra_and_property(
    G: Generator,
    latent_dim: int,
    n_samples: int,
    seq_len_spec: int,
    spec_min: float,
    spec_max: float,
    prop_min: float,
    prop_max: float,
    device: str = "cpu",
    out_path: str = "wgan_transformer_output/generated_pairs.xlsx"
):
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        gen_xy_norm = G(z)

    gen_xy_norm_np = gen_xy_norm.cpu().numpy()
    gen_spec_norm_np = gen_xy_norm_np[:, :seq_len_spec]
    gen_prop_norm_np = gen_xy_norm_np[:, -1:]

    spec_01 = (gen_spec_norm_np + 1.0) / 2.0
    spec_denorm = spec_01 * (spec_max - spec_min) + spec_min

    prop_01 = (gen_prop_norm_np + 1.0) / 2.0
    prop_denorm = prop_01 * (prop_max - prop_min) + prop_min

    band_cols = [f"band_{i+1}" for i in range(seq_len_spec)]

    df_den = pd.DataFrame(
        np.concatenate([spec_denorm, prop_denorm], axis=1),
        columns=band_cols + ["property"]
    )
    df_norm = pd.DataFrame(
        np.concatenate([gen_spec_norm_np, gen_prop_norm_np], axis=1),
        columns=band_cols + ["property_norm"]
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        df_den.to_excel(writer, sheet_name="denorm_pairs", index=False)
        df_norm.to_excel(writer, sheet_name="norm_-1_1_pairs", index=False)
    print(f"Saved generated spectra + property to: {out_path}")


if __name__ == "__main__":
    excel_path = r"xxx.xlsx"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 64

    G, D, (X_data, y_data, spec_min, spec_max, prop_min, prop_max, seq_len_spec) = train_wgan_transformer(
        excel_path=excel_path,
        device=device,
        latent_dim=latent_dim,
        batch_size=16,
        epochs=300,
        lr=1e-4,
        n_critic=5,
        out_dir="wgan_transformer_output"
    )

    export_attention_and_heatmaps(
        G,
        latent_dim=latent_dim,
        layer_idx=0,
        device=device,
        out_dir="wgan_transformer_output/attn_generator"
    )

    n_gen = X_data.shape[0]
    export_generated_spectra_and_property(
        G,
        latent_dim=latent_dim,
        n_samples=n_gen,
        seq_len_spec=seq_len_spec,
        spec_min=spec_min,
        spec_max=spec_max,
        prop_min=prop_min,
        prop_max=prop_max,
        device=device,
        out_path="wgan_transformer_output/generated_pairs.xlsx"
    )