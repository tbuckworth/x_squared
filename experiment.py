"""
Transformer vs MLP comparison on x^2 approximation.
Tests out-of-distribution generalization with Lion, Muon, Adam, and SGD optimizers.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

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


class Muon(torch.optim.Optimizer):
    """Muon optimizer - Momentum + Orthogonalization via Newton-Schulz."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz(G, steps=5):
        if G.ndim < 2:
            return G / (G.norm() + 1e-8)
        a, b, c = (3.4445, -4.7750, 2.0315)
        shape = G.shape
        if G.ndim > 2:
            G = G.reshape(G.shape[0], -1)
        X = G / (G.norm() + 1e-7)
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True
        else:
            transposed = False
        for _ in range(steps):
            A = X @ X.T
            B = a * torch.eye(A.shape[0], device=A.device, dtype=A.dtype) + b * A + c * A @ A
            X = B @ X
        if transposed:
            X = X.T
        return X.reshape(shape)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(self._newton_schulz(grad, ns_steps))
                if nesterov:
                    update = self._newton_schulz(grad, ns_steps) + momentum * buf
                else:
                    update = buf
                p.data.add_(update, alpha=-lr)
        return loss


# ============================================================
# Model Architectures
# ============================================================

class MLPModel(nn.Module):
    """MLP with 3 hidden layers, width 64."""
    def __init__(self, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x.unsqueeze(-1) if x.dim() == 1 else x)


class SmallTransformer(nn.Module):
    """
    Small Transformer for scalar regression.
    Embeds input scalar into d_model tokens, applies transformer encoder layers,
    then projects back to scalar output.
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


# ============================================================
# Training
# ============================================================

def make_optimizer(name, params, lr):
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=1e-3)
    elif name == 'SGD':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-3)
    elif name == 'Lion':
        return Lion(params, lr=lr * 0.1, weight_decay=1e-3)
    elif name == 'Muon':
        return Muon(params, lr=lr * 0.5)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, optimizer, x_train, y_train, x_test, y_test, epochs=3000, batch_size=256):
    device = next(model.parameters()).device
    loss_fn = nn.MSELoss()

    x_tr = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    x_te = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(-1)

    train_losses = []
    test_losses = []
    checkpoints = {}
    checkpoint_interval = max(epochs // 100, 1)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_tr.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, x_tr.size(0), batch_size):
            idx = perm[i:i+batch_size]
            bx, by = x_tr[idx], y_tr[idx]
            pred = model(bx)
            loss = loss_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        if epoch % checkpoint_interval == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                te_pred = model(x_te)
                te_loss = loss_fn(te_pred, y_te).item()
                test_losses.append((epoch, te_loss))
                checkpoints[epoch] = te_pred.cpu().numpy().squeeze()

        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                te_pred = model(x_te)
                te_loss = loss_fn(te_pred, y_te).item()
            print(f"  Epoch {epoch:5d} | Train Loss: {avg_loss:.2f} | Test Loss: {te_loss:.2f}", flush=True)

    return train_losses, test_losses, checkpoints


# ============================================================
# Visualization
# ============================================================

def plot_ood_comparison(x_test, y_test, all_results, train_low, train_high, out_path):
    architectures = ['MLP', 'Transformer']
    optimizers = ['Adam', 'SGD', 'Lion', 'Muon']

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(r'Out-of-Distribution Generalization: $y = x^2$' + '\nTransformer vs MLP across Optimizers',
                 fontsize=16, fontweight='bold')

    for i, arch in enumerate(architectures):
        for j, opt in enumerate(optimizers):
            ax = axes[i][j]
            key = f"{arch}_{opt}"
            if key not in all_results:
                ax.set_title(f"{arch} + {opt}\n(failed)")
                continue

            res = all_results[key]
            last_epoch = max(res['checkpoints'].keys())
            y_pred = res['checkpoints'][last_epoch]

            ax.axvspan(train_low, train_high, color='blue', alpha=0.15, label='Training Region')
            ax.scatter(x_test, y_test, color='grey', s=1, alpha=0.5, label=r'$y=x^2$')
            ax.scatter(x_test, y_pred, color='red', s=1, alpha=0.5, label='Prediction')
            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel('y', fontsize=11)
            final_test_loss = res['test_losses'][-1][1]
            ax.set_title(f"{arch} + {opt}\nTest Loss: {final_test_loss:.1f} | Params: {res['n_params']}",
                         fontsize=12)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='upper left')
            ax.set_ylim(-500, max(y_test) * 1.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_training_curves(all_results, out_path):
    architectures = ['MLP', 'Transformer']
    optimizers = ['Adam', 'SGD', 'Lion', 'Muon']

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle('Training & Test Loss Curves', fontsize=16, fontweight='bold')

    for i, arch in enumerate(architectures):
        for j, opt in enumerate(optimizers):
            ax = axes[i][j]
            key = f"{arch}_{opt}"
            if key not in all_results:
                ax.set_title(f"{arch} + {opt}\n(failed)")
                continue

            res = all_results[key]
            tl = res['train_losses']
            step = max(len(tl) // 500, 1)
            ax.plot(range(0, len(tl), step), tl[::step], color='blue', alpha=0.7, label='Train', linewidth=0.8)
            te_epochs = [e for e, _ in res['test_losses']]
            te_losses = [l for _, l in res['test_losses']]
            ax.plot(te_epochs, te_losses, color='red', alpha=0.9, label='Test (OOD)', linewidth=1.2)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('MSE Loss', fontsize=10)
            ax.set_yscale('log')
            ax.set_title(f"{arch} + {opt}", fontsize=12)
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_optimizer_comparison(all_results, out_path):
    architectures = ['MLP', 'Transformer']
    optimizers = ['Adam', 'SGD', 'Lion', 'Muon']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Final OOD Test Loss by Optimizer', fontsize=14, fontweight='bold')

    for i, arch in enumerate(architectures):
        ax = axes[i]
        losses = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for j, opt in enumerate(optimizers):
            key = f"{arch}_{opt}"
            if key in all_results:
                final_loss = all_results[key]['test_losses'][-1][1]
                losses.append(final_loss)
            else:
                losses.append(0)
        bars = ax.bar(optimizers, losses, color=colors)
        ax.set_title(f'{arch}', fontsize=13)
        ax.set_ylabel('MSE Loss (OOD)', fontsize=11)
        ax.set_yscale('log')
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{loss:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_extrapolation_detail(x_test, y_test, all_results, train_low, train_high, out_path):
    architectures = ['MLP', 'Transformer']
    optimizers = ['Adam', 'SGD', 'Lion', 'Muon']
    colors = {'Adam': '#1f77b4', 'SGD': '#ff7f0e', 'Lion': '#2ca02c', 'Muon': '#d62728'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Extrapolation Region Detail (Outside Training Range)',
                 fontsize=14, fontweight='bold')

    mask = (x_test < train_low) | (x_test > train_high)
    x_ood = x_test[mask]
    y_ood = y_test[mask]
    sort_idx = np.argsort(x_ood)
    x_ood_sorted = x_ood[sort_idx]
    y_ood_sorted = y_ood[sort_idx]

    for i, arch in enumerate(architectures):
        ax = axes[i]
        ax.plot(x_ood_sorted, y_ood_sorted, color='grey', linewidth=2, label=r'$y=x^2$', alpha=0.7)
        for opt in optimizers:
            key = f"{arch}_{opt}"
            if key not in all_results:
                continue
            last_epoch = max(all_results[key]['checkpoints'].keys())
            y_pred = all_results[key]['checkpoints'][last_epoch]
            y_pred_ood = y_pred[mask]
            y_pred_sorted = y_pred_ood[sort_idx]
            ax.plot(x_ood_sorted, y_pred_sorted, color=colors[opt], linewidth=1.5,
                    label=opt, alpha=0.8)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'{arch}', fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(-500, max(y_ood) * 1.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Report Generation
# ============================================================

def generate_report(all_results, out_path):
    architectures = ['MLP', 'Transformer']
    optimizers = ['Adam', 'SGD', 'Lion', 'Muon']

    lines = []
    lines.append("# Transformer vs MLP: Out-of-Distribution Generalization on y = x^2\n")
    lines.append("## Experiment Setup\n")
    lines.append("- **Task**: Approximate y = x^2 and test extrapolation beyond training range")
    lines.append("- **Training Range**: x in [-20, 20] (10,000 samples)")
    lines.append("- **Test Range**: x in [-50, 50] (10,000 samples)")
    lines.append("- **Epochs**: 3,000")
    lines.append("- **Batch Size**: 256")
    lines.append("- **Architectures**: MLP (3 hidden layers, width 64) vs Small Transformer (2 layers, d_model=32, 4 heads)")
    lines.append("- **Optimizers**: Adam, SGD, Lion, Muon\n")

    lines.append("## Model Sizes\n")
    lines.append("| Model | Parameters |")
    lines.append("|-------|-----------|")
    for key, res in all_results.items():
        lines.append(f"| {key} | {res['n_params']:,} |")
    lines.append("")

    lines.append("## Results Summary\n")
    lines.append("| Architecture | Optimizer | Final Train Loss | Final OOD Test Loss |")
    lines.append("|-------------|-----------|-----------------|-------------------|")
    for arch in architectures:
        for opt in optimizers:
            key = f"{arch}_{opt}"
            if key in all_results:
                r = all_results[key]
                train_l = r['train_losses'][-1]
                test_l = r['test_losses'][-1][1]
                lines.append(f"| {arch} | {opt} | {train_l:.2f} | {test_l:.2f} |")
    lines.append("")

    lines.append("## Visualizations\n")
    lines.append("### Final OOD Predictions\n")
    lines.append("![OOD Comparison](results/ood_comparison.png)\n")
    lines.append("### Training & Test Loss Curves\n")
    lines.append("![Loss Curves](results/loss_curves.png)\n")
    lines.append("### Optimizer Comparison (Final OOD Loss)\n")
    lines.append("![Optimizer Comparison](results/optimizer_comparison.png)\n")
    lines.append("### Extrapolation Region Detail\n")
    lines.append("![Extrapolation Detail](results/extrapolation_detail.png)\n")

    lines.append("## Analysis\n")

    best_key = None
    best_loss = float('inf')
    for key, res in all_results.items():
        tl = res['test_losses'][-1][1]
        if tl < best_loss:
            best_loss = tl
            best_key = key

    lines.append(f"**Best OOD Generalization**: {best_key} with test loss {best_loss:.2f}\n")

    for arch in architectures:
        arch_results = {k: v for k, v in all_results.items() if k.startswith(arch)}
        if arch_results:
            best_opt = min(arch_results, key=lambda k: arch_results[k]['test_losses'][-1][1])
            lines.append(f"- **{arch}**: Best optimizer is {best_opt.split('_')[1]} "
                        f"(OOD loss: {arch_results[best_opt]['test_losses'][-1][1]:.2f})")
    lines.append("")

    lines.append("### Optimizer Rankings (by OOD Test Loss)\n")
    for opt in optimizers:
        opt_results = {k: v for k, v in all_results.items() if k.endswith(opt)}
        if opt_results:
            avg_loss = np.mean([v['test_losses'][-1][1] for v in opt_results.values()])
            lines.append(f"- **{opt}**: Average OOD loss = {avg_loss:.2f}")
    lines.append("")

    lines.append("### Key Observations\n")
    lines.append("1. **In-distribution fit**: All optimizers can fit x^2 within the training range [-20, 20].")
    lines.append("2. **Extrapolation**: The critical test is how well each model/optimizer combo extends beyond [-20, 20].")
    lines.append("3. **Architecture effect**: Transformers and MLPs may learn fundamentally different representations,")
    lines.append("   leading to different extrapolation behaviors.")
    lines.append("4. **Optimizer effect**: Different optimizers find different local minima, which can dramatically")
    lines.append("   affect out-of-distribution behavior even when in-distribution performance is similar.\n")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(42)
    torch.manual_seed(42)
    train_low, train_high = -20, 20
    x_train = np.random.uniform(train_low, train_high, size=10000).astype(np.float32)
    y_train = (x_train ** 2).astype(np.float32)
    x_test = np.random.uniform(-50, 50, size=10000).astype(np.float32)
    y_test = (x_test ** 2).astype(np.float32)

    architectures = {
        'MLP': lambda: MLPModel(hidden=64, n_layers=3),
        'Transformer': lambda: SmallTransformer(d_model=32, nhead=4, num_layers=2, dim_ff=64, n_tokens=4),
    }
    optimizers_list = ['Adam', 'SGD', 'Lion', 'Muon']
    epochs = 3000
    lr = 1e-3

    all_results = {}

    for arch_name, make_model in architectures.items():
        for opt_name in optimizers_list:
            key = f"{arch_name}_{opt_name}"
            print(f"\n{'='*60}")
            print(f"Training: {key}")
            print(f"{'='*60}")

            torch.manual_seed(42)
            np.random.seed(42)
            model = make_model().to(device)
            n_params = count_parameters(model)
            print(f"  Parameters: {n_params:,}")

            optimizer = make_optimizer(opt_name, model.parameters(), lr)

            try:
                train_losses, test_losses, checkpoints = train_model(
                    model, optimizer, x_train, y_train, x_test, y_test,
                    epochs=epochs, batch_size=256
                )
                all_results[key] = {
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'checkpoints': checkpoints,
                    'n_params': n_params,
                }
                print(f"  Final Train Loss: {train_losses[-1]:.2f}")
                print(f"  Final Test Loss: {test_losses[-1][1]:.2f}")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    print("\n\nGenerating visualizations...")
    os.makedirs('results', exist_ok=True)

    plot_ood_comparison(x_test, y_test, all_results, train_low, train_high,
                       'results/ood_comparison.png')
    plot_training_curves(all_results, 'results/loss_curves.png')
    plot_optimizer_comparison(all_results, 'results/optimizer_comparison.png')
    plot_extrapolation_detail(x_test, y_test, all_results, train_low, train_high,
                             'results/extrapolation_detail.png')
    generate_report(all_results, 'REPORT.md')

    print("\nDone! Results saved to results/ and REPORT.md")


if __name__ == '__main__':
    main()
