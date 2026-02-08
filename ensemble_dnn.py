import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_bcr import BodyResponseDataset
from dataloader_ecg import *
from tqdm import tqdm
import os
import re
import numpy as np
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms

learning_rate = 1e-3
save_interval = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_checkpoint_path(args):
    base = f"./checkpoints/{args.data_type}"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "dnn_ensemble_checkpoint")

# =======================
# Args
# =======================
def get_args():
    parser = argparse.ArgumentParser(description="settings for DNN Ensemble Model")

    parser.add_argument("--data_type", type=str, choices=["bcr", "mnist", "ecg"], default="bcr")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="eval")
    parser.add_argument("--classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)

    parser.add_argument("--hidden_size", type=int, default=1)

    args = parser.parse_args()
    return args

def get_data_paths(args):
    data_type = args.data_type
    base = f"./dataset/{data_type}"
    if data_type == "mnist":
        return {"base": base}
    if data_type == "bcr":
        return {
            "base": base,
            "train": os.path.join(base, "bcr_train_cls9.xlsx"),         #trining data path
            "test": os.path.join(base, "bcr_test_cls9.xlsx"),           #testing data path
        }
    if data_type == "ecg":
        return {
            "base": base,
            "train": os.path.join(base, "ECG5000_TRAIN.ts"),
            "test": os.path.join(base, "ECG5000_TEST.ts"),
        }
    raise ValueError(f"Unknown data_type: {data_type}")

def build_dataset(args, split):
    if split not in ("train", "test"):
        raise ValueError(f"Unknown split: {split}")
    data_paths = get_data_paths(args)
    if args.data_type != "mnist":
        if not os.path.isdir(data_paths["base"]):
            raise FileNotFoundError(f"[DATA] Missing dataset folder: {data_paths['base']}")
        path = data_paths.get(split)
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"[DATA] Missing dataset file: {path}")

    if args.data_type == "bcr":
        path = data_paths[split]
        return BodyResponseDataset(path, classes=args.classes, transform=torch.tensor)
    if args.data_type == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        return datasets.MNIST(
            root=data_paths["base"],
            train=(split == "train"),
            download=True,
            transform=transform,
        )
    if args.data_type == "ecg":
        if split == "train":
            return ECGDataset(data_paths["train"], classes=None, transform=None)
        train_ds_tmp = ECGDataset(data_paths["train"], classes=None, transform=None)
        return ECGDataset(
            data_paths["test"],
            classes=None,
            norm_stats=train_ds_tmp.norm_stats,
            label_map=train_ds_tmp.targetMap,
            transform=None,
        )
    raise ValueError(f"Unknown data_type: {args.data_type}")

def setup_args(args):
    data_paths = get_data_paths(args)
    if args.data_type != "mnist":
        if not os.path.isdir(data_paths["base"]):
            raise FileNotFoundError(f"[DATA] Missing dataset folder: {data_paths['base']}")
        for key in ("train", "test"):
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
        args.classes = 9
    elif args.data_type == "mnist":
        args.classes = 10
    elif args.data_type == "ecg":
        # train 파일로부터 input_size/classes를 확정 (eval에서도 동일하게 맞춰야 checkpoint 로드가 됨)
        tmp = ECGDataset(data_paths["train"], classes=None, transform=None)
        args.classes = len(tmp.targetMap)
        assert tmp[0][0].numel() == args.input_size, (
            f"ECG input_size mismatch: expected {args.input_size}, "
            f"got {tmp[0][0].numel()}"
        )

    return args

# =======================
# DNN Module
# =======================
class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x

# =======================
# Accuracy
# =======================
def compute_accuracy(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.view(xb.size(0), -1)
            xb_cuda = xb.to(device)
            yb_cuda = yb.to(device)
            out = model(xb_cuda)
            loss = criterion(out, yb_cuda)
            total_loss += float(loss.item())
            _, predicted = torch.max(out.data, 1)
            total += int(yb_cuda.size(0))
            correct += int((predicted == yb_cuda).sum().item())

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy

# =======================
# Train
# =======================
def train(args):
    best_val_acc = []
    best_val_loss = []

    kfold = StratifiedKFold(n_splits=5)

    global checkpoint_path
    checkpoint_path = get_checkpoint_path(args)

    print(f"[TRAIN] Classes={args.classes}")

    dataset = build_dataset(args, "train")

    # stratified labels
    if args.data_type == "mnist":
        y_all = np.array(dataset.targets)
    else:  # bcr / ecg -> dataset.y
        y_all = np.array(dataset.y)
    y_all = y_all.astype(np.int64, copy=False)

    train_val_indices = np.arange(len(dataset))

    if args.data_type == "ecg":
        loop_interval = 1
        print_interval = 20
    else:
        loop_interval = 100
        print_interval = 100

    global_best_acc = -1.0
    global_best_epoch = None
    global_best_model_path = None
    global_best_model = None

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_val_indices, y_all)):
        print(f"[TRAIN] Fold {fold + 1}")
        print("[TRAIN] ------------------------------")
        best_epoch = None
        best_model_path = None
        best_model = None

        model = DNNModel(
            args.input_size,
            args.hidden_size,
            args.classes,
        )

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_subsampler,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=valid_subsampler,
            num_workers=4,
            pin_memory=True,
        )

        best_val_acc.append(0.0)
        best_val_loss.append(0.0)

        with tqdm(total=args.epochs, desc=f"Fold {fold + 1}", file=open("/dev/tty", "w")) as progress_bar:
            for epoch in range(args.epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.view(xb.size(0), -1)
                    xb_cuda = xb.to(device)
                    yb_cuda = yb.to(device)
                    out = model(xb_cuda)
                    loss = criterion(out, yb_cuda)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update(1)
                progress_bar.set_postfix(epoch=f"{epoch + 1}", loss=f"{loss.item():.4f}")

                if (epoch % loop_interval) == 0:
                    train_loss, train_pred_acc = compute_accuracy(model, train_loader)
                    valid_loss, val_pred_acc = compute_accuracy(model, val_loader)

                    if epoch % save_interval == 0:
                        if val_pred_acc > best_val_acc[fold]:
                            best_val_acc[fold] = val_pred_acc
                            best_val_loss[fold] = valid_loss
                            best_epoch = epoch
                            if max(best_val_acc) == best_val_acc[fold]:
                                best_model_path = (
                                    f"{checkpoint_path}_{args.classes}_{args.hidden_size}_{epoch}_FIN.pt"
                                )
                                best_model = model.state_dict()
                                if val_pred_acc > global_best_acc:
                                    global_best_acc = val_pred_acc
                                    global_best_epoch = epoch
                                    global_best_model_path = best_model_path
                                    global_best_model = best_model

                if (epoch % print_interval) == 0:
                    print(
                        f"[TRAIN] Epoch {epoch + 1}/{args.epochs} | "
                        f"Train Loss={train_loss:.4f}, Acc={train_pred_acc:.2%} | "
                        f"Val Loss={valid_loss:.4f}, Acc={val_pred_acc:.2%}"
                    )

        if best_model_path is None:
            best_model_path = (
                f"{checkpoint_path}_{args.classes}_{args.hidden_size}_{best_epoch}_FIN.pt"
            )
            best_model = model.state_dict()

        log_line = (
            f"Fold {fold + 1} | Best Epoch: {best_epoch} | "
            f"Val Acc: {best_val_acc[fold]:.4f} | Val Loss: {best_val_loss[fold]:.4f} | "
            f"Model Path: {best_model_path}\n"
        )

        print(f"[TRAIN] {log_line.strip()}")
        print(f"[TRAIN] {'-' * 20} FOLD {fold + 1} END {'-' * 20}\n")
        print(f"[TRAIN] Fold {fold + 1} Train FINISH")

    if global_best_model is None or global_best_model_path is None or global_best_epoch is None:
        raise RuntimeError("[TRAIN] No global best model selected. Check validation accuracy tracking.")
    torch.save(global_best_model, global_best_model_path)

    print("[EVAL] Evaluation BEST MODEL START")
    best_test_loss, best_test_acc = evaluate(args)

    log_line2 = (
        f"Global Best Epoch: {global_best_epoch} | "
        f"Test Acc: {best_test_acc:.4f} | Test Loss: {best_test_loss:.4f} | "
        f"Model Path: {global_best_model_path}\n"
    )

    print(f"[EVAL] {log_line2.strip()}")

# =======================
# Load target model
# =======================
def load_target_model(args):
    global checkpoint_path
    checkpoint_path = get_checkpoint_path(args)

    model = DNNModel(
        args.input_size,
        args.hidden_size,
        args.classes,
    )

    checkpoint_dir = os.path.dirname(checkpoint_path) or "."
    checkpoint_prefix = os.path.basename(checkpoint_path)

    pattern_fin = re.compile(
        rf"{re.escape(checkpoint_prefix)}_{args.classes}_{args.hidden_size}_(\d+)_FIN\.pt"
    )
    pattern = re.compile(
        rf"{re.escape(checkpoint_prefix)}_{args.classes}_{args.hidden_size}_(\d+)\.pt"
    )

    latest_epoch = -1
    latest_path = None

    for fname in os.listdir(checkpoint_dir):
        match = pattern_fin.fullmatch(fname)
        if match:
            ep = int(match.group(1))
            if ep > latest_epoch:
                latest_epoch = ep
                latest_path = os.path.join(checkpoint_dir, fname)

    if latest_path is None:
        for fname in os.listdir(checkpoint_dir):
            match = pattern.fullmatch(fname)
            if match:
                ep = int(match.group(1))
                if ep > latest_epoch:
                    latest_epoch = ep
                    latest_path = os.path.join(checkpoint_dir, fname)

    if latest_path:
        state = torch.load(latest_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        print(f"[CKPT] Load checkpoint complete: {latest_path}")
    else:
        print(f"[CKPT] No checkpoint found in: {checkpoint_dir}")
        raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")

    model.eval()

    model = model.to(device)
    return model

# =======================
# Evaluate
# =======================
def evaluate(args, input_loader=None):
    print(f"[EVAL] Classes={args.classes}")

    if input_loader is not None:
        test_loader = input_loader
    else:
        test_dataset = build_dataset(args, "test")

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    test_loss, test_pred_acc = compute_accuracy(load_target_model(args), test_loader)
    print(f"[EVAL] Acc={test_pred_acc:.2%}, Loss={test_loss:.4f}")

    return test_loss, test_pred_acc

# =======================
# Main
# =======================
if __name__ == "__main__":
    args = setup_args(get_args())

    checkpoint_path = get_checkpoint_path(args)
    mode = args.mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode not in ["train", "eval"]:
        print("Invalid mode. Please choose 'train' or 'eval'.")
        exit(1)

    if mode == "train":
        train(args)

    elif mode == "eval":
        evaluate(args)

    else:
        print("ERROR.")
