"""
Grokking experiment: Can a Transformer learning y=x^2 exhibit delayed generalization?

Based on insights from:
- Power et al. (2022) "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
- Liu et al. (2023) "Omnigrok: Grokking Beyond Algorithmic Data" (ICLR Spotlight)
- Lee et al. (2024) "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"

Key conditions for grokking:
1. Small training set (model can memorize easily)
2. High weight decay (pressure toward low-norm generalizing solutions)
3. Long training (10k-100k+ epochs)
4. Sufficient model capacity relative to dataset size

We test Transformer + Lion and Transformer + AdamW across weight decay settings,
with and without Grokfast gradient filtering, tracking train loss, in-distribution
test loss, OOD test loss, and weight norm evolution.
"""
import os
import gc
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_DIR = 'grokking_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Custom Optimizers
# ============================================================

class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al., 2023) - Evolved Sign Momentum."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(torch.sign(update), alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


# ============================================================
# Grokfast: Gradient Filter for Accelerated Grokking
# ============================================================

class GrokfastEMA:
    """
    Exponential moving average gradient filter from Grokfast (Lee et al., 2024).
    Amplifies slow-varying gradient components that correspond to generalizing updates.
    """
    def __init__(self, model, alpha=0.98, lamb=2.0):
        self.model = model
        self.alpha = alpha
        self.lamb = lamb
        self.ema_grads = {}

    def filter(self):
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if name not in self.ema_grads:
                self.ema_grads[name] = torch.zeros_like(p.grad)
            ema = self.ema_grads[name]
            ema.mul_(self.alpha).add_(p.grad, alpha=1 - self.alpha)
            p.grad.add_(ema, alpha=self.lamb)


# ============================================================
# Model Architecture
# ============================================================

class SmallTransformer(nn.Module):
    """
    Small Transformer for scalar regression.
    Same architecture as PR #1 to ensure continuity.
    """
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_ff=64, n_tokens=4):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.input_proj = nn.Linear(1, n_tokens * d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(n_tokens * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        B = x.shape[0]
        tokens = self.input_proj(x).reshape(B, self.n_tokens, self.d_model)
        tokens = tokens + self.pos_enc
        tokens = self.transformer(tokens)
        out = tokens.reshape(B, -1)
        return self.output_proj(out)

    def scale_init(self, factor):
        """Scale all parameters by factor (Omnigrok technique for inducing memorization)."""
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(factor)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weight_norm(model):
    """L2 norm of all parameters."""
    total = 0.0
    for p in model.parameters():
        total += p.data.norm(2).item() ** 2
    return math.sqrt(total)


# ============================================================
# Training
# ============================================================

def train_grokking(
    config_name, model, optimizer, x_train, y_train, x_test_id, y_test_id,
    x_test_ood, y_test_ood, epochs, grokfast=None, log_every=50,
    snapshot_epochs=None,
):
    """
    Train model for grokking detection.

    Returns dict with:
    - train_losses: list of (epoch, loss)
    - test_id_losses: list of (epoch, loss) -- in-distribution
    - test_ood_losses: list of (epoch, loss) -- out-of-distribution
    - weight_norms: list of (epoch, norm)
    - snapshots: dict of epoch -> predictions on OOD test set
    """
    device = next(model.parameters()).device
    loss_fn = nn.MSELoss()

    x_tr = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    x_id = torch.tensor(x_test_id, dtype=torch.float32, device=device)
    y_id = torch.tensor(y_test_id, dtype=torch.float32, device=device).unsqueeze(-1)
    x_ood = torch.tensor(x_test_ood, dtype=torch.float32, device=device)
    y_ood = torch.tensor(y_test_ood, dtype=torch.float32, device=device).unsqueeze(-1)

    if snapshot_epochs is None:
        snapshot_epochs = set()

    train_losses = []
    test_id_losses = []
    test_ood_losses = []
    weight_norms = []
    snapshots = {}

    start_time = time.time()
    last_print = start_time

    for epoch in range(epochs):
        # --- Train (full batch) ---
        model.train()
        pred = model(x_tr)
        loss = loss_fn(pred, y_tr)
        optimizer.zero_grad()
        loss.backward()

        if grokfast is not None:
            grokfast.filter()

        optimizer.step()

        # --- Log ---
        if epoch % log_every == 0 or epoch == epochs - 1 or epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                train_l = loss.item()
                id_pred = model(x_id)
                id_l = loss_fn(id_pred, y_id).item()
                ood_pred = model(x_ood)
                ood_l = loss_fn(ood_pred, y_ood).item()
                wn = weight_norm(model)

            train_losses.append((epoch, train_l))
            test_id_losses.append((epoch, id_l))
            test_ood_losses.append((epoch, ood_l))
            weight_norms.append((epoch, wn))

            if epoch in snapshot_epochs:
                snapshots[epoch] = ood_pred.cpu().numpy().squeeze()

        # --- Print progress (reuse logged values when available) ---
        now = time.time()
        if now - last_print > 30.0 or epoch == 0 or epoch == epochs - 1:
            elapsed = now - start_time
            eps = (epoch + 1) / elapsed if elapsed > 0 else 0
            eta = (epochs - epoch - 1) / eps if eps > 0 else 0
            tl = train_losses[-1][1] if train_losses else loss.item()
            il = test_id_losses[-1][1] if test_id_losses else 0
            print(
                f"  [{config_name}] Epoch {epoch:6d}/{epochs} | "
                f"Train: {tl:.4f} | ID Test: {il:.4f} | "
                f"Speed: {eps:.0f} ep/s | ETA: {eta:.0f}s",
                flush=True
            )
            last_print = now

    elapsed = time.time() - start_time
    print(f"  [{config_name}] Done in {elapsed:.1f}s ({epochs/elapsed:.0f} ep/s)")

    return {
        'train_losses': train_losses,
        'test_id_losses': test_id_losses,
        'test_ood_losses': test_ood_losses,
        'weight_norms': weight_norms,
        'snapshots': snapshots,
    }


# ============================================================
# Experiment Configurations
# ============================================================

def get_configs():
    """Define all experiment configurations."""
    return [
        # Lion at different weight decays
        {"name": "Lion_wd0.01", "opt": "Lion", "lr": 1e-4, "wd": 0.01,
         "init_scale": 1.0, "grokfast": False},
        {"name": "Lion_wd0.05", "opt": "Lion", "lr": 1e-4, "wd": 0.05,
         "init_scale": 1.0, "grokfast": False},
        {"name": "Lion_wd0.1", "opt": "Lion", "lr": 1e-4, "wd": 0.1,
         "init_scale": 1.0, "grokfast": False},

        # AdamW at different weight decays
        {"name": "AdamW_wd0.1", "opt": "AdamW", "lr": 1e-3, "wd": 0.1,
         "init_scale": 1.0, "grokfast": False},
        {"name": "AdamW_wd0.5", "opt": "AdamW", "lr": 1e-3, "wd": 0.5,
         "init_scale": 1.0, "grokfast": False},
        {"name": "AdamW_wd1.0", "opt": "AdamW", "lr": 1e-3, "wd": 1.0,
         "init_scale": 1.0, "grokfast": False},

        # Grokfast variants
        {"name": "Lion_wd0.05_grokfast", "opt": "Lion", "lr": 1e-4, "wd": 0.05,
         "init_scale": 1.0, "grokfast": True},
        {"name": "AdamW_wd0.5_grokfast", "opt": "AdamW", "lr": 1e-3, "wd": 0.5,
         "init_scale": 1.0, "grokfast": True},

        # High init scale (Omnigrok technique)
        {"name": "Lion_wd0.05_hi_init", "opt": "Lion", "lr": 1e-4, "wd": 0.05,
         "init_scale": 5.0, "grokfast": False},
        {"name": "AdamW_wd0.5_hi_init", "opt": "AdamW", "lr": 1e-3, "wd": 0.5,
         "init_scale": 5.0, "grokfast": False},
    ]


# ============================================================
# Visualization
# ============================================================

def plot_grokking_curves(all_results, out_path):
    """Main grokking plot: train vs in-distribution test loss for all configs."""
    n = len(all_results)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    fig.suptitle(
        r'Grokking Search: Transformer on $y = x^2$'
        '\nTrain Loss vs In-Distribution Test Loss (log scale)',
        fontsize=16, fontweight='bold', y=1.02
    )

    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx // cols][idx % cols]
        tr_epochs = [e for e, _ in res['train_losses']]
        tr_vals = [v for _, v in res['train_losses']]
        id_epochs = [e for e, _ in res['test_id_losses']]
        id_vals = [v for _, v in res['test_id_losses']]

        ax.plot(tr_epochs, tr_vals, color='blue', alpha=0.7, linewidth=0.8, label='Train')
        ax.plot(id_epochs, id_vals, color='red', alpha=0.9, linewidth=1.2, label='ID Test')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_weight_norms(all_results, out_path):
    """Weight norm evolution - key diagnostic for grokking."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Weight Norm Evolution During Training', fontsize=14, fontweight='bold')

    for name, res in all_results.items():
        epochs = [e for e, _ in res['weight_norms']]
        norms = [v for _, v in res['weight_norms']]
        ax.plot(epochs, norms, linewidth=1.2, label=name, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L2 Norm of Parameters', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_ood_evolution(all_results, out_path):
    """OOD test loss evolution."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(r'OOD Test Loss Evolution ($x \in [-50, 50]$)', fontsize=14, fontweight='bold')

    for name, res in all_results.items():
        epochs = [e for e, _ in res['test_ood_losses']]
        vals = [v for _, v in res['test_ood_losses']]
        ax.plot(epochs, vals, linewidth=1.2, label=name, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss (log)', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_prediction_snapshots(x_test_ood, y_test_ood, all_results, snapshot_epochs_list, out_path):
    """Prediction snapshots at key epochs for selected configs."""
    # Pick configs that have snapshots
    configs_with_snaps = {k: v for k, v in all_results.items() if v['snapshots']}
    if not configs_with_snaps:
        print("No snapshots to plot.")
        return

    n_configs = len(configs_with_snaps)
    n_snaps = len(snapshot_epochs_list)

    fig, axes = plt.subplots(n_configs, n_snaps, figsize=(4 * n_snaps, 4 * n_configs), squeeze=False)
    fig.suptitle(
        r'Prediction Evolution: Transformer on $y = x^2$'
        '\nSnapshots at Key Epochs',
        fontsize=14, fontweight='bold', y=1.02
    )

    sort_idx = np.argsort(x_test_ood)
    x_sorted = x_test_ood[sort_idx]
    y_sorted = y_test_ood[sort_idx]

    for i, (name, res) in enumerate(configs_with_snaps.items()):
        for j, ep in enumerate(snapshot_epochs_list):
            ax = axes[i][j]
            ax.plot(x_sorted, y_sorted, color='grey', linewidth=1, alpha=0.5, label=r'$y=x^2$')

            if ep in res['snapshots']:
                pred = res['snapshots'][ep][sort_idx]
                ax.plot(x_sorted, pred, color='red', linewidth=1, alpha=0.8, label='Pred')

            ax.set_ylim(-500, max(y_test_ood) * 1.1)
            ax.axvspan(-20, 20, color='blue', alpha=0.1)

            if i == 0:
                ax.set_title(f'Epoch {ep:,}', fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
            if i == n_configs - 1:
                ax.set_xlabel('x')
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_grokking_summary(all_results, out_path):
    """
    Summary plot: for each config, show the ratio of epochs-to-generalize / epochs-to-memorize.
    A high ratio indicates grokking. Also show final losses.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Grokking Summary', fontsize=14, fontweight='bold')

    names = []
    train_final = []
    id_final = []
    ood_final = []
    grok_ratios = []

    for name, res in all_results.items():
        names.append(name)
        train_final.append(res['train_losses'][-1][1])
        id_final.append(res['test_id_losses'][-1][1])
        ood_final.append(res['test_ood_losses'][-1][1])

        # Compute grokking ratio: epoch where ID test < threshold / epoch where train < threshold
        tr_vals = res['train_losses']
        id_vals = res['test_id_losses']

        # Use 1% of initial loss as threshold
        if len(tr_vals) > 0 and tr_vals[0][1] > 0:
            threshold = tr_vals[0][1] * 0.01
            train_converge = None
            for e, v in tr_vals:
                if v < threshold:
                    train_converge = e
                    break
            test_converge = None
            for e, v in id_vals:
                if v < threshold:
                    test_converge = e
                    break

            if train_converge and test_converge and train_converge > 0:
                grok_ratios.append(test_converge / train_converge)
            elif train_converge and not test_converge:
                grok_ratios.append(float('inf'))
            else:
                grok_ratios.append(1.0)
        else:
            grok_ratios.append(1.0)

    x_pos = np.arange(len(names))
    width = 0.25

    # Final losses
    ax1.bar(x_pos - width, train_final, width, label='Train', color='blue', alpha=0.7)
    ax1.bar(x_pos, id_final, width, label='ID Test', color='orange', alpha=0.7)
    ax1.bar(x_pos + width, ood_final, width, label='OOD Test', color='red', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Final MSE Loss')
    ax1.set_title('Final Losses by Configuration')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Grokking ratio
    finite_ratios = [r if r != float('inf') else 0 for r in grok_ratios]
    colors = ['green' if r > 5 else 'orange' if r > 2 else 'grey' for r in grok_ratios]
    bars = ax2.bar(x_pos, finite_ratios, color=colors, alpha=0.7)
    for i, r in enumerate(grok_ratios):
        if r == float('inf'):
            ax2.annotate('never\ngeneralized', (x_pos[i], max(finite_ratios) * 0.9 if finite_ratios else 1),
                        ha='center', fontsize=7, color='red')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Test Converge Epoch / Train Converge Epoch')
    ax2.set_title('Grokking Ratio (>5 = strong grokking)')
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Grokking threshold')
    ax2.axhline(y=1, color='grey', linestyle='--', alpha=0.5, label='No delay')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_train_vs_test_overlay(all_results, out_path):
    """Overlay all configs: train loss (dashed) vs ID test loss (solid) on single plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        r'Grokking Detection: Train vs Test Loss Overlay',
        fontsize=14, fontweight='bold'
    )

    # Split by optimizer type
    lion_configs = {k: v for k, v in all_results.items() if 'Lion' in k}
    adam_configs = {k: v for k, v in all_results.items() if 'Adam' in k}

    for ax, configs, title in [(ax1, lion_configs, 'Lion Configs'), (ax2, adam_configs, 'AdamW Configs')]:
        for name, res in configs.items():
            tr_epochs = [e for e, _ in res['train_losses']]
            tr_vals = [v for _, v in res['train_losses']]
            id_epochs = [e for e, _ in res['test_id_losses']]
            id_vals = [v for _, v in res['test_id_losses']]

            color = ax._get_lines.get_next_color()
            ax.plot(tr_epochs, tr_vals, color=color, linestyle='--', alpha=0.5,
                    linewidth=0.8, label=f'{name} (train)')
            ax.plot(id_epochs, id_vals, color=color, linestyle='-', alpha=0.9,
                    linewidth=1.2, label=f'{name} (ID test)')

        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def make_data():
    """Generate fixed train/test data."""
    np.random.seed(42)
    x_train = np.random.uniform(-20, 20, size=100).astype(np.float32)
    y_train = (x_train ** 2).astype(np.float32)
    x_test_id = np.random.uniform(-20, 20, size=500).astype(np.float32)
    y_test_id = (x_test_id ** 2).astype(np.float32)
    x_test_ood = np.random.uniform(-50, 50, size=2000).astype(np.float32)
    y_test_ood = (x_test_ood ** 2).astype(np.float32)
    return x_train, y_train, x_test_id, y_test_id, x_test_ood, y_test_ood


def run_single_config(cfg_index):
    """Run a single config and save results to its own JSON file."""
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = get_configs()
    cfg = configs[cfg_index]

    x_train, y_train, x_test_id, y_test_id, x_test_ood, y_test_ood = make_data()

    epochs = 100_000
    log_every = 200
    snapshot_epochs = {0, 500, 2500, 10000, 25000, 50000, 75000, 99999}

    print(f"Config [{cfg_index}]: {cfg['name']}")
    print(f"  {cfg['opt']}, lr={cfg['lr']}, wd={cfg['wd']}, "
          f"init_scale={cfg['init_scale']}, grokfast={cfg['grokfast']}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = SmallTransformer(d_model=32, nhead=4, num_layers=2, dim_ff=64, n_tokens=4).to(device)
    n_params = count_parameters(model)

    if cfg['init_scale'] != 1.0:
        model.scale_init(cfg['init_scale'])

    if cfg['opt'] == 'Lion':
        optimizer = Lion(model.parameters(), lr=cfg['lr'],
                       betas=(0.9, 0.99), weight_decay=cfg['wd'])
    elif cfg['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                     betas=(0.9, 0.98), weight_decay=cfg['wd'])

    grokfast_filter = GrokfastEMA(model, alpha=0.98, lamb=2.0) if cfg['grokfast'] else None

    result = train_grokking(
        config_name=cfg['name'], model=model, optimizer=optimizer,
        x_train=x_train, y_train=y_train,
        x_test_id=x_test_id, y_test_id=y_test_id,
        x_test_ood=x_test_ood, y_test_ood=y_test_ood,
        epochs=epochs, grokfast=grokfast_filter,
        log_every=log_every, snapshot_epochs=snapshot_epochs,
    )

    # Save individual result
    out = {
        'config': cfg, 'n_params': n_params,
        'train_losses': result['train_losses'],
        'test_id_losses': result['test_id_losses'],
        'test_ood_losses': result['test_ood_losses'],
        'weight_norms': result['weight_norms'],
        'snapshots': {str(k): v.tolist() for k, v in result['snapshots'].items()},
    }
    out_path = os.path.join(RESULTS_DIR, f'config_{cfg_index}_{cfg["name"]}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f"Saved: {out_path}")


def merge_and_plot():
    """Load all individual config results, merge, and generate plots."""
    import glob as globmod
    files = sorted(globmod.glob(os.path.join(RESULTS_DIR, 'config_*.json')))
    if not files:
        print("No results found!")
        return

    all_results = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        name = data['config']['name']
        # Convert snapshot keys back to int and values to numpy
        snaps = {}
        for k, v in data.get('snapshots', {}).items():
            snaps[int(k)] = np.array(v, dtype=np.float32)
        data['snapshots'] = snaps
        all_results[name] = data
        print(f"Loaded: {name}")

    # Save merged metrics
    metrics = {}
    for name, res in all_results.items():
        metrics[name] = {
            'config': res['config'], 'n_params': res['n_params'],
            'train_losses': res['train_losses'],
            'test_id_losses': res['test_id_losses'],
            'test_ood_losses': res['test_ood_losses'],
            'weight_norms': res['weight_norms'],
        }
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Recreate data for snapshot plots
    _, _, _, _, x_test_ood, y_test_ood = make_data()
    snapshot_epochs = {0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 99999}

    print("\nGenerating visualizations...")
    plot_grokking_curves(all_results, os.path.join(RESULTS_DIR, 'grokking_curves.png'))
    plot_weight_norms(all_results, os.path.join(RESULTS_DIR, 'weight_norms.png'))
    plot_ood_evolution(all_results, os.path.join(RESULTS_DIR, 'ood_evolution.png'))
    plot_prediction_snapshots(
        x_test_ood, y_test_ood, all_results, sorted(snapshot_epochs),
        os.path.join(RESULTS_DIR, 'prediction_snapshots.png'))
    plot_grokking_summary(all_results, os.path.join(RESULTS_DIR, 'grokking_summary.png'))
    plot_train_vs_test_overlay(all_results, os.path.join(RESULTS_DIR, 'train_vs_test_overlay.png'))

    # Summary table
    print("\n" + "="*70)
    print("GROKKING EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Train Loss':>12} {'ID Test':>12} {'OOD Test':>12} {'W.Norm':>10}")
    print("-"*76)
    for name, res in all_results.items():
        tl = res['train_losses'][-1][1]
        idl = res['test_id_losses'][-1][1]
        oodl = res['test_ood_losses'][-1][1]
        wn = res['weight_norms'][-1][1]
        print(f"{name:<30} {tl:>12.4f} {idl:>12.4f} {oodl:>12.1f} {wn:>10.2f}")
    print(f"\nResults saved to {RESULTS_DIR}/")


def main():
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'plot':
            merge_and_plot()
        else:
            run_single_config(int(sys.argv[1]))
    else:
        # Run all configs sequentially in separate subprocesses
        import subprocess
        configs = get_configs()
        print(f"Running {len(configs)} configs as separate processes...")
        for i, cfg in enumerate(configs):
            print(f"\n--- Starting config {i}: {cfg['name']} ---")
            result = subprocess.run(
                [sys.executable, __file__, str(i)],
                capture_output=False, timeout=900
            )
            if result.returncode != 0:
                print(f"  Config {i} ({cfg['name']}) failed with code {result.returncode}")
        # Generate plots from saved results
        merge_and_plot()


if __name__ == '__main__':
    main()
