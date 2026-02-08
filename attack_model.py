
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import math
import sys
from torchvision import datasets, transforms
from dataloader_bcr import BodyResponseDataset
from dataloader_ecg import ECGDataset
from ensemble_snn import load_target_model as load_target_model_snn
from ensemble_dnn import load_target_model as load_target_model_dnn

NOISE_VAL_LO = None
NOISE_VAL_HI = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser("SpikShield Attack Model")
    parser.add_argument("--data_type", type=str, choices=["bcr", "mnist", "ecg"], default="bcr", help="bcr or mnist or ecg")
    parser.add_argument("--input_size", type=int, default=None, help="input_size")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size (auto by data_type when None)")
    parser.add_argument("--epochs", type=int, default=None, help="epochs (auto by data_type when None)")

    parser.add_argument("--mode", type=str, choices=["train","eval"], default="eval", help="run mode: train or eval")
    parser.add_argument("--classes", type=int, default=None, help="number of classes")
    parser.add_argument("--num_steps", type=int, default=0, help="num_steps")
    parser.add_argument("--beta", type=float, default=0.9, help="beta")
    parser.add_argument("--hidden_size", type=int, default=20, help="hidden_size")

    parser.add_argument("--target", type=str, choices=["snn","dnn"], default="snn", help="target network type")
    parser.add_argument("--attack_hidden", type=int, default=512, help="attack MLP hidden dim")
    parser.add_argument("--attack_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--min_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=199)
    parser.add_argument("--query_size_at_once", type=int, default=5000)

    parser.add_argument("--candidates_per_class", type=int, default=500)
    parser.add_argument("--refine_steps", type=int, default=300)
    parser.add_argument("--refine_lr", type=float, default=5e-3)
    parser.add_argument("--refine_lam_prior", type=float, default=0.2)
    parser.add_argument("--lam_tv", type=float, default=1e-3)
    parser.add_argument("--retrain_after_refine", action="store_true", help="retrain surrogate with refined dataset (default: False)")
    parser.add_argument("--prior_sigma", type=float, default=0.3, help="prior noise sigma")

    parser.add_argument("--max_query_num", type=int, default=20000)

    parser.add_argument("--val_p_lo", type=float, default=1.0, help="lower value percentile")
    parser.add_argument("--val_p_hi", type=float, default=99.0, help="upper value percentile")

    parser.add_argument("--noise_amp_alpha", type=float, default=0.3, help="delta = alpha * (val_hi - val_lo)")
    parser.add_argument("--external_eval_path", type=str, default="./attack_datasets/")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_attack", help="surrogate checkpoint directory")

    return parser.parse_args()

def get_data_paths(args):
    data_type = args.data_type
    base = f"./dataset/{data_type}"
    if data_type == "mnist":
        return {"base": base}
    if data_type == "bcr":
        return {
            "base": base,
            "prior": os.path.join(base, "bcr.xlsx"),
            "test": os.path.join(base, "bcr_test_cls9.xlsx"),
            "trained_target": os.path.join(base, "bcr_train_cls9.xlsx"),
        }
    if data_type == "ecg":
        return {
            "base": base,
            "trained_target": os.path.join(base, "ECG5000_TRAIN.ts"),
            "test": os.path.join(base, "ECG5000_TEST.ts"),
        }
    raise ValueError(f"Unknown data_type: {data_type}")

def setup_args(args):
    data_paths = get_data_paths(args)
    if args.data_type != "mnist":
        if not os.path.isdir(data_paths["base"]):
            raise FileNotFoundError(f"[DATA] Missing dataset folder: {data_paths['base']}")
        keys = ("trained_target", "test")
        if args.data_type == "bcr":
            keys = ("prior", "test", "trained_target")
        for key in keys:
            path = data_paths.get(key)
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"[DATA] Missing dataset file: {path}")

    expected_input_sizes = {
        "bcr": 380,
        "mnist": 784,
        "ecg": 140,
    }
    expected_input_size = expected_input_sizes[args.data_type]
    if args.input_size is None:
        args.input_size = expected_input_size
    else:
        assert args.input_size == expected_input_size, (
            f"input_size mismatch for {args.data_type}: "
            f"expected {expected_input_size}, got {args.input_size}"
        )

    if args.data_type == "bcr":
        expected_classes = 9
        if args.classes is None:
            args.classes = expected_classes
        else:
            assert args.classes == expected_classes, (
                f"classes mismatch for bcr: expected {expected_classes}, got {args.classes}"
            )
        if args.batch_size is None:
            args.batch_size = 300
        if args.epochs is None:
            args.epochs = 2000
    elif args.data_type == "mnist":
        expected_classes = 10
        if args.classes is None:
            args.classes = expected_classes
        else:
            assert args.classes == expected_classes, (
                f"classes mismatch for mnist: expected {expected_classes}, got {args.classes}"
            )
        if args.batch_size is None:
            args.batch_size = 2048
        if args.epochs is None:
            args.epochs = 400
        args.prior_sigma = 0.1
    elif args.data_type == "ecg":
        expected_classes = 5
        if args.classes is None:
            args.classes = expected_classes
        else:
            assert args.classes == expected_classes, (
                f"classes mismatch for ecg: expected {expected_classes}, got {args.classes}"
            )
        if args.batch_size is None:
            args.batch_size = 300
        if args.epochs is None:
            args.epochs = 200

    return args

def make_attack_ckpt_name(args):
    if args.target == "snn":
        return f"attack_model_{args.data_type}_{args.target}_{args.num_steps}_{args.hidden_size}.pt"
    else:
        return f"attack_model_{args.data_type}_{args.target}_{args.hidden_size}.pt"

def cosine_sim_vec(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:

    a = a.view(-1).float()
    b = b.view(-1).float()
    denom = (a.norm(p=2) * b.norm(p=2)).clamp(min=eps)
    return float(torch.dot(a, b) / denom)

class SurrogateNet(nn.Module):
    def __init__(self, input_dim=380, num_classes=10, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def lowpass_filter_tensor(x, keep_ratio=0.2):
    Xf = torch.fft.rfft(x, dim=1)
    cutoff = max(1, int(Xf.shape[1] * keep_ratio))
    Xf[:, cutoff:] = 0
    xr = torch.fft.irfft(Xf, n=x.shape[1], dim=1)
    return xr

def total_variation_loss(x: torch.Tensor, data_type: str):
    if data_type == "mnist":

        x2 = x.view(-1, 1, 28, 28)
        tv_h = (x2[:, :, 1:, :] - x2[:, :, :-1, :]).abs().mean()
        tv_w = (x2[:, :, :, 1:] - x2[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w
    else:

        return (x[:, 1:] - x[:, :-1]).abs().mean()

def refine_input_via_gradient(
    args,
    surrogate,
    x_init,
    target_class,
    prior_mean=None,
    steps=300,
    lr=1e-2,
    lam_prior=1.0,
    lam_tv=1e-3,
    lowpass_keep=0.3,
    device="cpu",
    noise_amp_alpha=0.3,
    range_lo=None,
    range_hi=None,
    save_sample_idx=10
):
    x = x_init.clone().detach().to(device)
    x.requires_grad_(True)

    optim_x = torch.optim.Adam([x], lr=lr)
    surrogate.eval()

    p_vec = None
    if prior_mean is not None:
        if args.data_type == "bcr":
            p_vec = prior_mean.view(1, -1).to(device)

    delta = None
    if (range_lo is not None) and (range_hi is not None):
        span = max(1e-8, float(range_hi) - float(range_lo))
        delta = max(1e-8, float(noise_amp_alpha) * span)

    for t in range(steps):
        optim_x.zero_grad()

        logits = surrogate(x)
        loss_cls = -logits[:, target_class].mean()

        loss_prior = 0.0
        if (p_vec is not None) and (lam_prior is not None) and (lam_prior != 0):
            loss_prior = float(lam_prior) * ((x - p_vec) ** 2).mean()

        loss_tv = 0.0
        if (lam_tv is not None) and (lam_tv != 0):
            loss_tv = float(lam_tv) * total_variation_loss(x, args.data_type)

        loss = loss_cls + loss_prior + loss_tv
        loss.backward()
        optim_x.step()

        if (lowpass_keep is not None) and (t % 50 == 0):
            with torch.no_grad():
                x.data = lowpass_filter_tensor(x.data, keep_ratio=lowpass_keep).to(device)

        if args.data_type == "mnist":
            with torch.no_grad():
                x.data.clamp_(0.0, 1.0)
        elif args.data_type == "bcr":
            if (p_vec is not None) and (delta is not None):
                with torch.no_grad():
                    eps = x.data - p_vec
                    max_abs, _ = eps.abs().max(dim=1, keepdim=True)
                    scale = torch.ones_like(max_abs)
                    mask = max_abs > delta
                    scale[mask] = delta / (max_abs[mask] + 1e-12)
                    x.data = p_vec + eps * scale

            if (range_lo is not None) and (range_hi is not None):
                with torch.no_grad():
                    x_min, _ = x.data.min(dim=1, keepdim=True)
                    x_max, _ = x.data.max(dim=1, keepdim=True)
                    need = (x_min < range_lo) | (x_max > range_hi)
                    if need.any():
                        denom = (x_max - x_min).clamp(min=1e-8)
                        scale = (range_hi - range_lo) / denom
                        x.data = (x.data - x_min) * scale + range_lo

        elif args.data_type == "ecg":
            with torch.no_grad():
                x.data = (x.data - x.data.mean(dim=1, keepdim=True)) / (x.data.std(dim=1, keepdim=True).clamp_min(1e-6))
                x.data.clamp_(-5.0, 5.0)

    return x.detach().cpu()

def evaluate_target(pool, labels, target, X_ref, y_ref, num_classes, device="cpu", batch_size=256, eval_data=0):
    target.eval()
    pool = pool.to(device)
    labels = labels.to(device)
    total = 0
    correct = 0
    tot_c = [0 for _ in range(num_classes)]
    corr_c = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for i in range(0, pool.size(0), batch_size):
            xb = pool[i:i+batch_size]
            yb = labels[i:i+batch_size]

            logits = target(xb)
            preds = logits.argmax(dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

            yb_cpu = yb.cpu()
            preds_cpu = preds.cpu()
            for c in range(num_classes):
                mask = (yb_cpu == c)
                if mask.any():
                    tot_c[c]  += mask.sum().item()
                    corr_c[c] += (preds_cpu[mask] == yb_cpu[mask]).sum().item()

    acc = correct / max(1, total)

    acc_per_class = []
    for c in range(num_classes):
        if tot_c[c] > 0:
            acc_per_class.append(corr_c[c] / tot_c[c])
        else:
            acc_per_class.append(float("nan"))

    class_means = {}
    X_ref = X_ref.to(device).float()
    y_ref = y_ref.to(device).long()
    for c in range(num_classes):
        idx = (y_ref == c).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        class_means[c] = X_ref[idx].mean(dim=0)

    mse_list = []
    psnr_list = []

    cos_list = []
    cos_per_class = []

    for c in range(num_classes):
        idx_ref = (y_ref == c).nonzero(as_tuple=False).squeeze(-1)
        if idx_ref.numel() == 0:
            continue
        ref_c = X_ref[idx_ref]
        ref_mean = ref_c.mean(dim=0)

        idx_gen = (labels == c).nonzero(as_tuple=False).squeeze(-1)
        if idx_gen.numel() == 0:

            continue
        gen_c = pool[idx_gen]
        gen_mean = gen_c.mean(dim=0)

        mse_c = F.mse_loss(gen_mean, ref_mean, reduction='mean').item()
        mse_list.append(mse_c)

        max_ref = ref_mean.abs().max().item()
        max_gen = gen_mean.abs().max().item()
        maxv = max(max_ref, max_gen)

        psnr_c = 20.0 * math.log10(
            (maxv + 1e-8) / math.sqrt(mse_c + 1e-8)
        )
        psnr_list.append(psnr_c)
        cos_c = cosine_sim_vec(gen_mean, ref_mean)
        cos_list.append(cos_c)

    mean_mse = float(np.mean(mse_list)) if mse_list else float("nan")
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else float("nan")
    mean_cos = float(np.mean(cos_list)) if cos_list else float("nan")

    print(f"[evaluate_acc of target network](with Test set of target model) : "
        f"Acc={acc:.4f}, MSE(class-mean vs class-mean)={mean_mse:.6f}, "
        f"PSNR(class-mean vs class-mean)={mean_psnr:.2f} dB, "
        f"Cosine(class-mean vs class-mean)={mean_cos:.4f}")

    print("per-class acc:")
    for c in range(num_classes):
        v = acc_per_class[c]
        if math.isnan(v):
            print(f"  class {c}: NaN (no samples)")
        else:
            print(f"  class {c}: {v:.4f}")

    return {
        "acc": acc,
        "mse": mean_mse,
        "psnr": mean_psnr,
        "cos": mean_cos,
        "acc_per_class": acc_per_class,
    }

def evaluate_surrogate_fidelity(surrogate, target, ext_x, ext_y, device="cpu", batch_size=256, eval_data=0):
    surrogate.eval(); target.eval()
    tot, agree = 0, 0

    num_classes = int(ext_y.max().item()) + 1
    tot_c = torch.zeros(num_classes, dtype=torch.long)
    agree_c = torch.zeros(num_classes, dtype=torch.long)

    sur_tot = 0
    sur_correct = 0
    sur_tot_c = torch.zeros(num_classes, dtype=torch.long)
    sur_correct_c = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for i in range(0, ext_x.size(0), batch_size):
            xb = ext_x[i:i+batch_size].to(device)
            yb = ext_y[i:i+batch_size].to(device)

            pred_t = target(xb).argmax(1)
            pred_s = surrogate(xb).argmax(1)

            matches = (pred_t == pred_s)

            agree += matches.sum().item()
            tot += yb.size(0)

            for c in range(num_classes):
                mask_c = (yb == c)
                if mask_c.any():
                    tot_c[c] += mask_c.sum().item()
                    agree_c[c] += (matches & mask_c).sum().item()

            sur_matches = (pred_s == yb)
            sur_correct += sur_matches.sum().item()
            sur_tot += yb.size(0)

            for c in range(num_classes):
                mask_c = (yb == c)
                if mask_c.any():
                    sur_tot_c[c] += mask_c.sum().item()
                    sur_correct_c[c] += (sur_matches & mask_c).sum().item()

    acc = agree / max(1, tot)

    fidelity_per_class = []
    for c in range(num_classes):
        if tot_c[c] > 0:
            fidelity_per_class.append(agree_c[c].item() / tot_c[c].item())
        else:
            fidelity_per_class.append(float("nan"))

    sur_acc = sur_correct / max(1, sur_tot)

    sur_acc_per_class = []
    for c in range(num_classes):
        if sur_tot_c[c] > 0:
            sur_acc_per_class.append(sur_correct_c[c].item() / sur_tot_c[c].item())
        else:
            sur_acc_per_class.append(float("nan"))

    if eval_data == 0:
        print(f"[Fidelity](with generated testset during attack) : {acc:.4f}")
    elif eval_data == 1:
        print(f"[Fidelity](with Test set of target model) : {acc:.4f}")

    print("[fidelity per class]")
    for c in range(num_classes):
        v = fidelity_per_class[c]
        print(f"  class {c}: {fidelity_per_class[c]:.4f}" if not math.isnan(fidelity_per_class[c]) else
              f"  class {c}: NaN (no samples)")

    if eval_data == 1:
        print(f"[Surrogate accuracy](with Test set of target model): {sur_acc:.4f}")
        print("[accuracy per class]")
        for c in range(num_classes):
            v = sur_acc_per_class[c]
            print(f"  class {c}: {v:.4f}" if not math.isnan(v) else
                f"  class {c}: NaN (no samples)")

    return acc, fidelity_per_class, sur_acc, sur_acc_per_class

def get_refined_xlsx_path(args):
    if args.target == "snn":
        return f"./attack_datasets/attack_refined_{args.data_type}_{args.target}_{args.num_steps}_{args.hidden_size}.xlsx"
    return f"./attack_datasets/attack_refined_{args.data_type}_{args.target}_{args.hidden_size}.xlsx"

def get_step0_xlsx_path(args):
    if args.target == "snn":
        return f"./attack_datasets/attack_step0_{args.data_type}_{args.target}_{args.num_steps}_{args.hidden_size}.xlsx"
    return f"./attack_datasets/attack_step0_{args.data_type}_{args.target}_{args.hidden_size}.xlsx"

def load_refined_from_xlsx(args, device="cpu"):
    path = get_refined_xlsx_path(args)

    if not os.path.exists(path):
        raise FileNotFoundError(f"[load_refined_from_xlsx] file not found: {path}")

    df = pd.read_excel(path)
    refined_pool_labels = torch.tensor(df["Response"].values, dtype=torch.long, device=device)
    refined_pool = torch.tensor(df.drop(columns=["index", "Response"]).values, dtype=torch.float32, device=device)

    print(f"[*] refined_pool loaded -> {path} | X={tuple(refined_pool.shape)}, y={tuple(refined_pool_labels.shape)}")
    return refined_pool, refined_pool_labels

def load_step0_from_xlsx(args, device="cpu"):
    path = get_step0_xlsx_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[load_step0_from_xlsx] file not found: {path}")
    df = pd.read_excel(path)
    step0_labels = torch.tensor(df["Response"].values, dtype=torch.long, device=device)
    step0_pool = torch.tensor(df.drop(columns=["index", "Response"]).values, dtype=torch.float32, device=device)
    print(f"[*] step0_pool loaded -> {path} | X={tuple(step0_pool.shape)}, y={tuple(step0_labels.shape)}")
    return step0_pool, step0_labels

def thief_data(args, target, device, input_dim):
    global NOISE_VAL_LO, NOISE_VAL_HI
    data_paths = get_data_paths(args)
    template_path = data_paths["prior"] if args.data_type == "bcr" else None
    query_size_at_once = args.query_size_at_once
    max_query_num = args.max_query_num
    num_batches = max_query_num // query_size_at_once

    X_thief = np.empty((max_query_num, input_dim), dtype=np.float32)
    y_thief = np.empty((max_query_num,), dtype=np.int64)

    target.eval()

    if args.data_type == "bcr":
        prior_ds = BodyResponseDataset(template_path, transform=torch.tensor, prior_labels=[2, 7, 11, 13, 19, 20])
        prior_mean = prior_ds.get_mean_processed()
        X_prior_all = torch.tensor(prior_ds.x, dtype=torch.float32)

        if isinstance(prior_mean, np.ndarray):
            prior_mean = torch.from_numpy(prior_mean).to(device).float()
        else:
            prior_mean = prior_mean.to(device).float()
        x_np = X_prior_all.cpu().numpy()
        val_lo = float(np.percentile(x_np, args.val_p_lo))
        val_hi = float(np.percentile(x_np, args.val_p_hi))
        span = max(1e-12, val_hi - val_lo)
        noise_delta = max(1e-8, args.noise_amp_alpha * span)
        NOISE_VAL_LO, NOISE_VAL_HI = val_lo, val_hi

        for batch_idx in range(num_batches):
            prior_m = prior_mean.unsqueeze(0).repeat(query_size_at_once, 1)
            noise = torch.empty_like(prior_m, device=device).uniform_(-noise_delta, noise_delta)
            x_t = prior_m + noise

            x_min, _ = x_t.min(dim=1, keepdim=True)
            x_max, _ = x_t.max(dim=1, keepdim=True)
            denom = (x_max - x_min).clamp(min=1e-12)
            scale = (val_hi - val_lo) / denom
            x_t = (x_t - x_min) * scale + val_lo

            with torch.no_grad():
                y_pred = target(x_t).argmax(1).detach().cpu().numpy().astype(np.int64)

            start = batch_idx * query_size_at_once
            end = start + query_size_at_once
            X_thief[start:end] = x_t.detach().cpu().numpy()
            y_thief[start:end] = y_pred

    elif args.data_type == "mnist":
        for batch_idx in range(num_batches):

            x_t = torch.sigmoid(torch.randn((query_size_at_once, input_dim), device=device) * 1.0 - 3.0)

            with torch.no_grad():
                y_pred = target(x_t).argmax(1).detach().cpu().numpy().astype(np.int64)

            start = batch_idx * query_size_at_once
            end = start + query_size_at_once

            X_thief[start:end] = x_t.detach().cpu().numpy()
            y_thief[start:end] = y_pred

    elif args.data_type == "ecg":
        for batch_idx in range(num_batches):

            x_t = torch.empty((query_size_at_once, input_dim), device=device).uniform_(-1.0, 1.0)
            with torch.no_grad():
                y_pred = target(x_t).argmax(1).detach().cpu().numpy().astype(np.int64)

            start = batch_idx * query_size_at_once
            end = start + query_size_at_once

            X_thief[start:end] = x_t.detach().cpu().numpy()
            y_thief[start:end] = y_pred

    print()
    dist = np.bincount(y_thief, minlength=args.classes).tolist()
    print(f"Class dist: {dist}")
    X_thief = torch.as_tensor(X_thief).float()
    y_thief = torch.as_tensor(y_thief).long()
    return X_thief, y_thief

def Data_external(args, target, device, input_dim):
    global NOISE_VAL_LO, NOISE_VAL_HI

    os.makedirs("./attack_datasets", exist_ok=True)
    thief_data_path = args.external_eval_path
    query_size_at_once_ex = 5000
    data_paths = get_data_paths(args)
    template_path = data_paths["prior"] if args.data_type == "bcr" else None

    if args.data_type == "bcr":
        if args.target == "snn":
            thief_data_path += f"{args.data_type}_{args.target}_ns{args.num_steps}_hs{args.hidden_size}"
        else:
            thief_data_path += f"{args.data_type}_{args.target}_hs{args.hidden_size}"
    elif args.data_type == "mnist":
        if args.target == "snn":
            thief_data_path += f"{args.data_type}_{args.target}_ns{args.num_steps}_hs{args.hidden_size}_thief"
        else:
            thief_data_path += f"{args.data_type}_{args.target}_hs{args.hidden_size}_thief"
    elif args.data_type == "ecg":
        if args.target == "snn":
            thief_data_path += f"{args.data_type}_{args.target}_ns{args.num_steps}_hs{args.hidden_size}_ecg_thief"
        else:
            thief_data_path += f"{args.data_type}_{args.target}_hs{args.hidden_size}_ecg_thief"

    npz_path = thief_data_path + ".npz"

    if os.path.exists(npz_path):
        cache = np.load(npz_path)
        X_thief = cache["X"].astype(np.float32)
        y_thief = cache["y"].astype(np.int64)

        if args.data_type == "bcr":
            NOISE_VAL_LO = float(np.percentile(X_thief, args.val_p_lo))
            NOISE_VAL_HI = float(np.percentile(X_thief, args.val_p_hi))
        X_thief_t = torch.from_numpy(X_thief).float()
        y_thief_t = torch.from_numpy(y_thief).long()
        return X_thief_t, y_thief_t

    iter = 0
    per_class = 5000
    counts = np.zeros(args.classes, dtype=np.int64)

    if args.data_type == "bcr":
        prior_ds = BodyResponseDataset(template_path, transform=torch.tensor, prior_labels=[2, 7, 11, 13, 19, 20])
        prior_mean = prior_ds.get_mean_processed()
        X_prior_all = torch.tensor(prior_ds.x, dtype=torch.float32)

        if isinstance(prior_mean, np.ndarray):
            prior_mean = torch.from_numpy(prior_mean).to(device).float()
        else:
            prior_mean = prior_mean.to(device).float()
        x_np = X_prior_all.cpu().numpy()
        val_lo = float(np.percentile(x_np, args.val_p_lo))
        val_hi = float(np.percentile(x_np, args.val_p_hi))
        span = max(1e-12, val_hi - val_lo)
        noise_delta = max(1e-8, args.noise_amp_alpha * span)
        NOISE_VAL_LO, NOISE_VAL_HI = val_lo, val_hi

    X_thief = np.empty((args.classes * per_class, input_dim), dtype=np.float32)
    y_thief = np.empty((args.classes * per_class,), dtype=np.int64)

    target.eval()
    use_seed_next = False
    seed_x = [np.empty((0, input_dim), dtype=np.float32) for _ in range(args.classes)]

    while counts.min() < per_class:
        iter += 1
        missing_before = int(np.maximum(0, per_class - counts).sum())

        if not use_seed_next:

            if args.data_type == "bcr":
                prior_m = prior_mean.unsqueeze(0).repeat(query_size_at_once_ex, 1)
                noise = torch.empty_like(prior_m, device=device).uniform_(-noise_delta, noise_delta)

                x_t = prior_m + noise
                if (NOISE_VAL_LO is not None) and (NOISE_VAL_HI is not None):
                    lo = NOISE_VAL_LO
                    hi = NOISE_VAL_HI

                    x_min, _ = x_t.min(dim=1, keepdim=True)
                    x_max, _ = x_t.max(dim=1, keepdim=True)
                    denom = (x_max - x_min).clamp(min=1e-12)
                    scale = (hi - lo) / denom
                    x_t = (x_t - x_min) * scale + lo

            elif args.data_type == "mnist":
                x_t = torch.sigmoid(torch.randn((query_size_at_once_ex, input_dim), device=device) * 1.0 - 3.0)
            elif args.data_type == "ecg":

                x_t = torch.empty((query_size_at_once_ex, input_dim), device=device).uniform_(-1.0, 1.0)
        else:
            pools = [seed_x[c] for c in range(args.classes) if (counts[c] < per_class) and (seed_x[c].shape[0] > 0)]
            if len(pools) == 0:
                if args.data_type == "bcr":
                    prior_m = prior_mean.unsqueeze(0).repeat(query_size_at_once_ex, 1)
                    noise = torch.empty_like(prior_m, device=device).uniform_(-noise_delta, noise_delta)

                    x_t = prior_m + noise

                    if (NOISE_VAL_LO is not None) and (NOISE_VAL_HI is not None):
                        lo = NOISE_VAL_LO
                        hi = NOISE_VAL_HI

                        x_min, _ = x_t.min(dim=1, keepdim=True)
                        x_max, _ = x_t.max(dim=1, keepdim=True)
                        denom = (x_max - x_min).clamp(min=1e-12)
                        scale = (hi - lo) / denom
                        x_t = (x_t - x_min) * scale + lo

                elif args.data_type == "mnist":
                    x_t = torch.sigmoid(torch.randn((query_size_at_once_ex, input_dim), device=device) * 1.0 - 3.0)
                elif args.data_type == "ecg":

                    x_t = torch.empty((query_size_at_once_ex, input_dim), device=device).uniform_(-1.0, 1.0)
            else:
                pool_all = np.concatenate(pools, axis=0)
                pick = np.random.randint(0, pool_all.shape[0], size=query_size_at_once_ex)
                selected_pool = torch.from_numpy(pool_all[pick]).to(device)
                eps = 0.05
                neg = torch.minimum(torch.full_like(selected_pool, eps), selected_pool)
                pos = torch.minimum(torch.full_like(selected_pool, eps), 1.0 - selected_pool)
                u = torch.rand_like(selected_pool)
                noise = -neg + u * (pos + neg)
                x_t = selected_pool + noise

        with torch.no_grad():
            pred = target(x_t).argmax(1)

        x_np = x_t.detach().cpu().numpy()
        for c in range(args.classes):
            need = per_class - counts[c]
            if need <= 0:
                continue

            idx = (pred == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if idx.numel() > need:
                idx = idx[:need]

            k = int(idx.numel())
            take = x_np[idx.detach().cpu().numpy()]

            base = c * per_class
            s = base + counts[c]
            e = s + k

            X_thief[s:e] = take
            y_thief[s:e] = c
            counts[c] += k

            if seed_x[c].shape[0] == 0:
                seed_x[c] = take
            else:
                seed_x[c] = np.concatenate([seed_x[c], take], axis=0)
                if seed_x[c].shape[0] > per_class:
                    seed_x[c] = seed_x[c][-per_class:]

            if counts[c] >= per_class:
                seed_x[c] = np.empty((0, input_dim), dtype=np.float32)

        missing_after = int(np.maximum(0, per_class - counts).sum())
        use_seed_next = (missing_after == missing_before) and (missing_after > 0)

        if iter % 100 == 0 or missing_after == 0:
            perc = 100.0 * counts / per_class
            msg = f"[build_pool] iter={iter} missing={missing_after} | " +\
                  " ".join(f"{ci}:{p:5.1f}%" for ci, p in enumerate(perc))
            print(msg, end="\r", flush=True)

    np.savez(npz_path, X=X_thief, y=y_thief)
    X_thief_t = torch.from_numpy(X_thief).float()
    y_thief_t = torch.from_numpy(y_thief).long()
    return X_thief_t, y_thief_t

def train_surrogate(surrogate, args, device, train_x, train_y, val_x, val_y):
    opt = torch.optim.Adam(surrogate.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    bs = args.batch_size

    dl_tr = DataLoader(TensorDataset(train_x, train_y), batch_size=bs, shuffle=True)
    dl_va = DataLoader(TensorDataset(val_x, val_y), batch_size=bs, shuffle=False)

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for ep in range(1, args.attack_epochs + 1):
        surrogate.train()
        run_loss = 0.0
        ns = 0

        for xb, yb in dl_tr:
            xb = xb.to(device).float()
            yb = yb.to(device).long()

            opt.zero_grad()
            logits = surrogate(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            run_loss += loss.item() * xb.size(0)
            ns += xb.size(0)

        surrogate.eval()
        correct = 0
        tot = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device).float()
                yb = yb.to(device).long()
                logits = surrogate(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                tot += yb.size(0)

        val_acc = correct / max(1, tot)

        if (ep % 10 == 0) or (ep == 1):
            print(f"[surrogate][thief] {ep}/{args.attack_epochs} train_loss={run_loss/max(1,ns):.6f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in surrogate.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (ep >= args.min_epochs) and (patience_counter >= args.patience):
            print(f"[Early Stop] epoch={ep}, best_val_acc={best_val_acc:.4f}")
            break

    if best_state is not None:
        surrogate.load_state_dict(best_state)

def split_train_val(X, y, val_ratio, seed):
    ds = TensorDataset(X.float(), y.long())
    n_val = int(len(ds) * val_ratio)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = torch.utils.data.random_split(
        ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    tr_x, tr_y = tr_ds[:]
    va_x, va_y = va_ds[:]
    return tr_x, tr_y, va_x, va_y

def evaluate_pool(pool, labels, X_ref, y_ref, num_classes, device="cpu"):
    pool = pool.to(device)
    labels = labels.to(device)

    class_means = {}
    X_ref = X_ref.to(device).float()
    y_ref = y_ref.to(device).long()
    for c in range(num_classes):
        idx = (y_ref == c).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        class_means[c] = X_ref[idx].mean(dim=0)

    mse_list = []
    psnr_list = []
    cos_list = []

    for c in range(num_classes):
        idx_ref = (y_ref == c).nonzero(as_tuple=False).squeeze(-1)
        if idx_ref.numel() == 0:
            continue
        ref_c = X_ref[idx_ref]
        ref_mean = ref_c.mean(dim=0)

        idx_gen = (labels == c).nonzero(as_tuple=False).squeeze(-1)
        if idx_gen.numel() == 0:

            continue
        gen_c = pool[idx_gen]
        gen_mean = gen_c.mean(dim=0)

        mse_c = F.mse_loss(gen_mean, ref_mean, reduction='mean').item()
        mse_list.append(mse_c)

        max_ref = ref_mean.abs().max().item()
        max_gen = gen_mean.abs().max().item()
        maxv = max(max_ref, max_gen)

        psnr_c = 20.0 * math.log10(
            (maxv + 1e-8) / math.sqrt(mse_c + 1e-8)
        )
        psnr_list.append(psnr_c)
        cos_c = cosine_sim_vec(gen_mean, ref_mean)
        cos_list.append(cos_c)
    mean_mse = float(np.mean(mse_list)) if mse_list else float("nan")
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else float("nan")
    mean_cos = float(np.mean(cos_list)) if cos_list else float("nan")

    return {
        "mse": mean_mse,
        "psnr": mean_psnr,
        "cos": mean_cos,
        "psnr_per_class": psnr_list,
    }

def main():
    args = setup_args(get_args())
    data_paths = get_data_paths(args)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    print(f"Device={device}, CUDA={torch.cuda.device_count()}")

    if args.data_type == "bcr":
        bcr_prior_path = data_paths["prior"]
        bcr_test_path = data_paths["test"]
        bcr_trained_target = data_paths["trained_target"]
    elif args.data_type == "ecg":
        ecg_trained_path = data_paths["trained_target"]
        ecg_test_path = data_paths["test"]

    target = load_target_model_dnn(args) if args.target == "dnn" else load_target_model_snn(args)
    target.eval()

    if args.target == "snn":
        print("Target: SNN")
    elif args.target == "dnn":
        print("Target: DNN")
    args.seed = int(time.time() * 1000) % (2**32 - 1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[thief_data seed] {args.seed}")
    print("[*] Preparing external eval set (cache-enabled)...")
    external_eval_t, labels_ext_t = Data_external(args, target, device, args.input_size)
    print(f"External eval set: {external_eval_t.shape}, class dist={torch.bincount(labels_ext_t, minlength=args.classes).tolist()}")

    if args.data_type == "bcr":
        target_eval = BodyResponseDataset(bcr_test_path, classes=args.classes, transform=torch.tensor)
        ext_x_list = []
        ext_y_list = []

        for i in range(len(target_eval)):
            x_i, y_i = target_eval[i]
            ext_x_list.append(x_i.unsqueeze(0))
            ext_y_list.append(int(y_i.item()))

        original_evalset_x = torch.cat(ext_x_list, dim=0).float()
        original_evalset_y = torch.tensor(ext_y_list, dtype=torch.long)

        ds = BodyResponseDataset(bcr_trained_target, classes=args.classes, transform=torch.tensor)

        original_trainset_x = torch.from_numpy(np.asarray(ds.x, dtype=np.float32))
        original_trainset_y = torch.from_numpy(np.asarray(ds.y, dtype=np.int64))
        prior_ds = BodyResponseDataset(bcr_prior_path, transform=torch.tensor, prior_labels=[2, 7, 11, 13, 19, 20])
        X_prior_all = torch.tensor(prior_ds.x, dtype=torch.float32); y_prior_all = torch.tensor(prior_ds.y, dtype=torch.long)
        prior_mean = prior_ds.get_mean_processed()

        if isinstance(prior_mean, np.ndarray):
            prior_mean = torch.from_numpy(prior_mean).to(device).float()
        else:
            prior_mean = prior_mean.to(device).float()

        print(f"[ATTACKER DATA] X={X_prior_all.shape}, classes(in prior set)={int(y_prior_all.max().item())+1}")
        print(f"Loaded original train: X={original_trainset_x.shape}, classes={args.classes}")

    elif args.data_type == "mnist":
        transform = transforms.ToTensor()
        mnist_test = datasets.MNIST(root=data_paths["base"], train=False, download=True, transform=transform)
        original_evalset_x = mnist_test.data.view(-1, 28*28).float() / 255.0
        original_evalset_y = mnist_test.targets.long()
        transform = transforms.ToTensor()
        mnist_train = datasets.MNIST(root=data_paths["base"], train=True, download=True, transform=transform)
        original_trainset_x = mnist_train.data.view(-1, 28*28).float() / 255.0
        original_trainset_y = mnist_train.targets.long()
    elif args.data_type == "ecg":
        transform = transforms.ToTensor()
        target_eval = ECGDataset(ecg_test_path, classes=None, transform=None)
        ext_x_list = []
        ext_y_list = []

        for i in range(len(target_eval)):
            x_i, y_i = target_eval[i]
            ext_x_list.append(x_i.unsqueeze(0))
            ext_y_list.append(int(y_i.item()))

        original_evalset_x = torch.cat(ext_x_list, dim=0).float()
        original_evalset_y = torch.tensor(ext_y_list, dtype=torch.long)

        ds = ECGDataset(ecg_trained_path, classes=None, transform=None)

        original_trainset_x = torch.from_numpy(np.asarray(ds.x, dtype=np.float32))
        original_trainset_y = torch.from_numpy(np.asarray(ds.y, dtype=np.int64))
        print(f"Loaded original train: X={original_trainset_x.shape}, classes={args.classes}")

    target_eval_test = evaluate_target(original_evalset_x, original_evalset_y, target, original_trainset_x, original_trainset_y, args.classes, device=device, eval_data=1)

    if args.mode == "eval":
        surrogate = SurrogateNet(input_dim=args.input_size, num_classes=args.classes, hidden=args.attack_hidden).to(device)
        ckpt_root = os.path.join(args.ckpt_dir, "refined") if args.retrain_after_refine > 0 else args.ckpt_dir
        if not os.path.isdir(args.ckpt_dir):
            raise FileNotFoundError(f"[ERROR] eval mode but checkpoint dir missing: {args.ckpt_dir}")

        ckpt_path = os.path.join(ckpt_root, make_attack_ckpt_name(args))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[ERROR] eval mode: checkpoint not found for current settings.\n"
                f"  data_type={args.data_type}, "
                f"  target={args.target}, num_steps={args.num_steps}, "
                f"  hidden_size={args.hidden_size}\n"
                f"  expected file: {ckpt_path}"
            )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        surrogate.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
        print(f"[*] Loaded attacker surrogate checkpoint -> {ckpt_path}")

        fidelity_acc, fidelity_per_class, sur_acc, sur_acc_per_class = evaluate_surrogate_fidelity(surrogate, target, external_eval_t, labels_ext_t, device=device, eval_data=0)

        fidelity_acc_target_eval, fidelity_per_class_target_eval, sur_acc_target_eval, sur_acc_per_class_target_eval = evaluate_surrogate_fidelity(surrogate, target, original_evalset_x, original_evalset_y, device=device, eval_data=1)

    if args.mode == "train":

        pool, pool_labels = thief_data(args, target, device, args.input_size)
        print("Rough pool:", pool.shape, " class dist:", np.bincount(pool_labels.numpy(), minlength=args.classes).tolist())

        surrogate = SurrogateNet(input_dim=args.input_size, num_classes=args.classes, hidden=args.attack_hidden).to(device)

        X, y = pool.float(), pool_labels.long()
        tr_x, tr_y, val_x, val_y = split_train_val(X, y, args.val_frac, args.seed)
        print(f"Synth train/val: {len(tr_y)}/{len(val_y)}")

        print("[*] Training surrogate on rough dataset ...")
        train_surrogate(surrogate, args, device, tr_x, tr_y, val_x, val_y)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        save_path = os.path.join(args.ckpt_dir, make_attack_ckpt_name(args))
        torch.save({"state_dict": surrogate.state_dict()}, save_path)
        print(f"[*] Saved surrogate checkpoint -> {save_path}")
        results_round = evaluate_pool(tr_x, tr_y, original_evalset_x, original_evalset_y, args.classes, device=device)
        nan_metrics = {"mse": float("nan"), "psnr": float("nan"), "cos": float("nan")}
        results_step300 = nan_metrics
        os.makedirs("./attack_datasets", exist_ok=True)
        step0_path = get_step0_xlsx_path(args)
        df_step0 = pd.DataFrame(tr_x.cpu().numpy())
        df_step0.insert(0, "Response", tr_y.cpu().numpy().astype(int))
        df_step0.insert(0, "index", range(len(df_step0)))
        df_step0.to_excel(step0_path, index=False)
        print(f"[*] step0_pool saved -> {step0_path}")

        if args.data_type == "bcr":
            print("\n======== (3) Refinement ========")
            print(f"\n[Refinement] -----------------------")
            surrogate.eval()
            for p in surrogate.parameters(): p.requires_grad_(False)
            refined_list, refined_labels = [], []
            for c in range(args.classes):
                idx = (tr_y == c).nonzero(as_tuple=True)[0]
                if idx.numel() == 0: continue
                idx = idx[:args.candidates_per_class]

                x0 = tr_x[idx].to(device).float()

                x_refined = refine_input_via_gradient(
                    args=args,
                    surrogate=surrogate,
                    x_init=x0,
                    target_class=c,
                    prior_mean=prior_mean if args.data_type == "bcr" else None,
                    steps=args.refine_steps,
                    lr=args.refine_lr,
                    lam_prior=args.refine_lam_prior,
                    lam_tv=args.lam_tv,
                    lowpass_keep=0.3,
                    device=device,
                    noise_amp_alpha=args.noise_amp_alpha,
                    range_lo=NOISE_VAL_LO,
                    range_hi=NOISE_VAL_HI
                )

                refined_list.append(x_refined.detach().cpu())
                refined_labels.append(torch.full((x_refined.size(0),), c, dtype=torch.long))

            refined_x = torch.cat(refined_list, 0)
            refined_y = torch.cat(refined_labels, 0)
            print("Refined:", refined_x.shape, "dist:", torch.bincount(refined_y, minlength=args.classes).tolist())

            if args.data_type == "bcr":
                save_path = get_refined_xlsx_path(args)
                df_refined = pd.DataFrame(refined_x.numpy())
                df_refined.insert(0, "Response", refined_y.numpy().astype(int))
                df_refined.insert(0, "index", range(len(df_refined)))
                df_refined.to_excel(save_path, index=False)
                print(f"[*] refined_pool saved -> {save_path}")

            fidelity_acc, fidelity_per_class, sur_acc, sur_acc_per_class = evaluate_surrogate_fidelity(surrogate, target, external_eval_t, labels_ext_t, device=device, eval_data=0)

            fidelity_acc_target_eval, fidelity_per_class_target_eval, sur_acc_target_eval, sur_acc_per_class_target_eval = evaluate_surrogate_fidelity(surrogate, target, original_evalset_x, original_evalset_y, device=device, eval_data=1)

            print(f"[evaluate_target] evaluating target ({args.target}) ...")

            results_step300 = evaluate_pool(refined_x, refined_y, original_evalset_x, original_evalset_y, args.classes, device=device)

            print(f"[Step300][target={args.target}] Eval Results: {results_step300}")

            pass

        else:

            fidelity_acc, fidelity_per_class, sur_acc, sur_acc_per_class = evaluate_surrogate_fidelity(surrogate, target, external_eval_t, labels_ext_t, device=device, eval_data=0)

            fidelity_acc_target_eval, fidelity_per_class_target_eval, sur_acc_target_eval, sur_acc_per_class_target_eval = evaluate_surrogate_fidelity(surrogate, target, original_evalset_x, original_evalset_y, device=device, eval_data=1)
    else:
        nan_metrics = {"mse": float("nan"), "psnr": float("nan"), "cos": float("nan")}
        results_round = nan_metrics
        results_step300 = nan_metrics
        try:
            step0_pool, step0_labels = load_step0_from_xlsx(args, device=device)
            results_round = evaluate_pool(step0_pool, step0_labels, original_evalset_x, original_evalset_y, args.classes, device=device)
        except FileNotFoundError as e:
            print(f"[EVAL] step0 file missing: {e}")
        if args.data_type == "bcr":
            refined_pool, refined_labels = load_refined_from_xlsx(args, device=device)
            results_step300 = evaluate_pool(refined_pool, refined_labels, original_evalset_x, original_evalset_y, args.classes, device=device)
            print(f"[EVAL] Refined eval results: {results_step300}")

    try:
        if args.mode != "eval":
            print("[INFO] Skip summary CSV generation in train mode.")
            print("Done.")
            return

        base_path = f"./eval_{args.data_type}_attack_results_summary_snn" if args.target == "snn" else f"./eval_{args.data_type}_attack_results_summary_dnn"
        base_path += f"_round{args.refine_steps}" if args.data_type == "bcr" else ""
        summary_path = f"{base_path}.csv"
        logs = {
            "target": args.target,
            "num_steps": args.num_steps,
            "hidden_size": args.hidden_size,
            "refined": 1 if args.data_type == "bcr" else 0,
            "mse_step0": float(results_round["mse"]) if not isinstance(results_round["mse"], float) else results_round["mse"],
            "psnr_step0": float(results_round["psnr"]) if not isinstance(results_round["psnr"], float) else results_round["psnr"],
            "cos_step0": float(results_round.get("cos", float("nan"))),
            "mse_step300": float(results_step300["mse"]) if not isinstance(results_step300["mse"], float) else results_step300["mse"],
            "psnr_step300": float(results_step300["psnr"]) if not isinstance(results_step300["psnr"], float) else results_step300["psnr"],
            "cos_step300": float(results_step300.get("cos", float("nan"))),

            "acc_target_eval": target_eval_test["acc"],
            "mse_target_eval": float(target_eval_test["mse"]) if not isinstance(target_eval_test["mse"], float) else target_eval_test["mse"],
            "psnr_target_eval": float(target_eval_test["psnr"]) if not isinstance(target_eval_test["psnr"], float) else target_eval_test["psnr"],
            "fidelity_acc": fidelity_acc,
            "fidelity_acc_target_eval": fidelity_acc_target_eval,
            "sur_acc": sur_acc,
            "sur_acc_target_eval": sur_acc_target_eval,

            "n_samples": int(args.max_query_num),
        }

        acc_per_class_target_eval = target_eval_test.get("acc_per_class", [])

        for c in range(args.classes):

            if c < len(acc_per_class_target_eval):
                logs[f"acc_per_class{c}"] = acc_per_class_target_eval[c]
            else:
                logs[f"acc_per_class{c}"] = float("nan")
            if c < len(acc_per_class_target_eval):
                logs[f"acc_per_class_target_eval{c}"] = acc_per_class_target_eval[c]
            else:
                logs[f"acc_per_class_target_eval{c}"] = float("nan")

        for c in range(args.classes):
            if c < len(fidelity_per_class):
                logs[f"fidelity_per_class{c}"] = fidelity_per_class[c]
            else:
                logs[f"fidelity_per_class{c}"] = float("nan")

        for c in range(args.classes):
            if c < len(fidelity_per_class_target_eval):
                logs[f"fidelity_per_class_target_eval{c}"] = fidelity_per_class_target_eval[c]
            else:
                logs[f"fidelity_per_class_target_eval{c}"] = float("nan")

        for c in range(args.classes):
            if c < len(sur_acc_per_class):
                logs[f"sur_acc_per_class{c}"] = sur_acc_per_class[c]
            else:
                logs[f"sur_acc_per_class{c}"] = float("nan")
        for c in range(args.classes):
            if c < len(sur_acc_per_class_target_eval):
                logs[f"sur_acc_per_class_target_eval{c}"] = sur_acc_per_class_target_eval[c]
            else:
                logs[f"sur_acc_per_class_target_eval{c}"] = float("nan")

        if os.path.exists(summary_path):
            df_prev = pd.read_csv(summary_path)
            df_prev = pd.concat([df_prev, pd.DataFrame([logs])], ignore_index=True)
            df_prev.to_csv(summary_path, index=False)
        else:
            pd.DataFrame([logs]).to_csv(summary_path, index=False)

        print(f"Saved unified summary -> {summary_path}")
    except Exception as e:
        print("Failed to save summary:", e)

    print("Done.")

if __name__ == "__main__":
    main()
