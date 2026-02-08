# dataloader_ecg.py
import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
  def __init__(self, path, classes=None, norm_stats=None, label_map=None, transform=None):
    self.transform = transform

    # ===== 1) 파일 파싱 (.ts) =====
    x_list = []
    y_raw_list = []
    in_data = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
      for line_raw in f:
        line = line_raw.strip()
        if not line or line.startswith("#"):
          continue
        if not in_data:
          if line.lower().startswith("@data"):
            in_data = True
          continue

        parts = line.split(":")
        label_str = parts[-1].strip()
        channel_strs = parts[:-1]

        channels = []
        for ch_str in channel_strs:
          vals = []
          for v in ch_str.split(","):
            v = v.strip()
            if v == "" or v == "?":
              vals.append(0.0)
            else:
              vals.append(float(v))
          channels.append(vals)

        sample = np.asarray(channels, dtype=np.float32)  # [C, L]
        x_list.append(sample)
        y_raw_list.append(label_str)

    # 라벨 int로
    y_int = [int(float(s)) for s in y_raw_list]

    # classes 필터 (classes가 리스트일 때만)
    if classes is not None and isinstance(classes, (list, tuple, set)):
      keep_idx = [i for i, lab in enumerate(y_int) if lab in classes]
      x_list = [x_list[i] for i in keep_idx]
      y_int  = [y_int[i] for i in keep_idx]

    # ===== 2) 라벨 매핑 (train에서 만든 label_map을 test에 그대로 주입) =====
    if label_map is None:
      self.target = sorted(list(set(y_int)))
      self.targetMap = {t: i for i, t in enumerate(self.target)}
    else:
      self.targetMap = dict(label_map)
      self.target = sorted(self.targetMap.keys())

    self.y = [self.targetMap[lab] for lab in y_int]  # 0..K-1
    self.x = x_list
    self.dataSize = len(self.x)

    # ===== 3) 정규화 (train에서 mean/std 계산 → test에 주입) =====
    if norm_stats is None:
      all_x = np.concatenate([arr.reshape(-1) for arr in self.x], axis=0).astype(np.float32)
      mean = float(all_x.mean())
      std  = float(all_x.std() + 1e-8)
      self.norm_stats = {"mean": mean, "std": std}
    else:
      mean = float(norm_stats["mean"])
      std  = float(norm_stats["std"]) if float(norm_stats["std"]) > 1e-8 else 1e-8
      self.norm_stats = {"mean": mean, "std": std}

    mean = self.norm_stats["mean"]
    std  = self.norm_stats["std"]
    self.x = [((arr - mean) / std).astype(np.float32) for arr in self.x]

  def __len__(self):
    return self.dataSize

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    x = torch.tensor(self.x[idx], dtype=torch.float32)   # [C, L] or [L]
    y = torch.tensor(self.y[idx], dtype=torch.long)

    if x.dim() == 2:
        if x.size(0) == 1:
            x = x.squeeze(0)  # [L]
        else:
            x = x[0]

    if self.transform:
        x = self.transform(x)

    return x, y
