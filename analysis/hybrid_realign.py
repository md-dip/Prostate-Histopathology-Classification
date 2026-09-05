"""
Hybrid Alignment Script (Standalone)
-------------------------------------
Re-infers Hybrid predictions on the SAME 901-sample test set that
LGCN / GAT / JGCN / JGCN-matched share, so cross-model McNemar can run.

Why this script exists:
  The Hybrid notebook used random.shuffle + slice for train/val/test.
  The other 4 notebooks used sklearn.train_test_split(stratify, random_state=42).
  These two produce different test sets. To pair Hybrid predictions with
  the baselines, Hybrid needs to run inference on the baselines' test set.

What it does:
  1. Rebuilds the Hybrid model class and loads best_hybrid_model_v2.pth
  2. Loads Hybrid's SLIC-superpixel graph cache
  3. Rebuilds the baselines' test set using the EXACT same algorithm
     the LGCN/GAT/JGCN notebooks use (Path.iterdir + train_test_split
     with stratify=all_labels, random_state=42, 70/15/15)
  4. Runs one non-TTA inference pass on those 901 samples
  5. Saves hybrid_aligned_raw.npz with file_paths + y_true + y_pred + y_prob

Run:   python hybrid_realign.py
Time:  ~2 minutes on RTX 3060, no training.

After it finishes:
       python mcnemar_aligned.py
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split

# ═════════════════════════════════════════════════════════════════════
# Paths & constants (match Hybrid notebook + LGCN notebook exactly)
# ═════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------
# Paths resolve through the repository-level config module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
# ---------------------------------------------------------------------

DATA_DIR         = str(C.DATA_DIR)
# Build this cache first:  python preprocessing/build_slic_graph_cache.py
GRAPH_CACHE_PATH = str(C.GRAPH_CACHE_HYBRID)
MODEL_PATH       = str(C.CHECKPOINT_DIR / 'hybrid_best.pth')
OUT_PATH         = str(C.HYBRID_DIR / 'hybrid_aligned_raw.npz')

SEED       = C.SEED
IMG_SIZE   = C.IMG_SIZE
BATCH_SIZE = C.BATCH_SIZE
HIDDEN     = C.HIDDEN_DIM
GAT_HEADS  = C.GAT_HEADS
DROPOUT    = C.DROPOUT
IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp'}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Determinism
import random
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ═════════════════════════════════════════════════════════════════════
# Hybrid Model — verbatim copy of Cell 5 in Proposed_Hybrid_Prostate_.ipynb
# ═════════════════════════════════════════════════════════════════════
class HybridProstateCancerNet(nn.Module):
    def __init__(self, num_classes=2, hidden=256, gat_heads=4, dropout=0.4):
        super().__init__()
        H = hidden

        # Branch 1: EfficientNet-B4 backbone
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False,   # no download; state_dict loads it
            num_classes=0, global_pool='avg'
        )
        cnn_dim = self.backbone.num_features   # 1792

        # Freeze early layers (matches training, safe to leave for eval)
        blocks_to_freeze = ['blocks.0', 'blocks.1', 'blocks.2',
                            'blocks.3', 'conv_stem', 'bn1']
        for name, param in self.backbone.named_parameters():
            if any(b in name for b in blocks_to_freeze):
                param.requires_grad = False

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Branch 2: GAT
        self.gat_node_proj = nn.Sequential(nn.Linear(cnn_dim, H), nn.ReLU())
        self.gat1 = GATConv(H, H // gat_heads, heads=gat_heads, dropout=0.2)
        self.gat2 = GATConv(H, H // gat_heads, heads=gat_heads, dropout=0.2)
        self.gat_norm1 = nn.LayerNorm(H)
        self.gat_norm2 = nn.LayerNorm(H)

        # Branch 3: 3-layer GCN on SLIC superpixels
        self.gcn_input_proj = nn.Sequential(nn.Linear(8, H), nn.ReLU())
        self.gcn1 = GCNConv(H,      H)
        self.gcn2 = GCNConv(H,      H // 2)
        self.gcn3 = GCNConv(H // 2, H // 4)
        self.gcn_norm1 = nn.LayerNorm(H)
        self.gcn_norm2 = nn.LayerNorm(H // 2)
        self.gcn_norm3 = nn.LayerNorm(H // 4)

        # Fusion — 512 + 256 + 64 = 832
        fusion_dim = 512 + H + H // 4
        self.se_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 8),
            nn.ReLU(),
            nn.Linear(fusion_dim // 8, fusion_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, imgs, graph_batch):
        B = imgs.size(0)
        cnn_feats = self.backbone(imgs)
        cnn_out   = self.cnn_proj(cnn_feats)

        # GAT (fully connected within batch, drop self-loops)
        gat_nodes = self.gat_node_proj(cnn_feats)
        src = torch.arange(B, device=imgs.device).repeat_interleave(B)
        dst = torch.arange(B, device=imgs.device).repeat(B)
        mask   = src != dst
        ei_gat = torch.stack([src[mask], dst[mask]])
        bv_gat = torch.arange(B, device=imgs.device)

        x = self.gat_norm1(F.elu(self.gat1(gat_nodes, ei_gat)))
        x = self.gat_norm2(F.elu(self.gat2(x,        ei_gat)))
        gat_out = global_mean_pool(x, bv_gat)

        # GCN
        if graph_batch is not None:
            gx = self.gcn_input_proj(graph_batch.x)
            ei = graph_batch.edge_index
            gb = graph_batch.batch
            gx = self.gcn_norm1(F.relu(self.gcn1(gx, ei)))
            gx = self.gcn_norm2(F.relu(self.gcn2(gx, ei)))
            gx = self.gcn_norm3(F.relu(self.gcn3(gx, ei)))
            gcn_out = global_mean_pool(gx, gb)
        else:
            gcn_out = torch.zeros(B, self.gcn_norm3.normalized_shape[0],
                                  device=imgs.device)

        fused = torch.cat([cnn_out, gat_out, gcn_out], dim=1)
        gate  = self.se_gate(fused)
        fused = fused * gate
        return self.classifier(fused)

# ═════════════════════════════════════════════════════════════════════
# Dataset — verbatim from Hybrid notebook
# ═════════════════════════════════════════════════════════════════════
class ProstateDataset(Dataset):
    def __init__(self, file_list, transform, graph_cache):
        self.files       = file_list
        self.transform   = transform
        self.graph_cache = graph_cache

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        img_tensor  = self.transform(Image.open(path).convert('RGB'))
        cached      = self.graph_cache.get(path)
        if cached is not None:
            graph = Data(
                x          = torch.tensor(cached['x'],          dtype=torch.float),
                edge_index = torch.tensor(cached['edge_index'], dtype=torch.long),
                y          = torch.tensor(label,                dtype=torch.long)
            )
        else:
            graph = None
        return img_tensor, torch.tensor(label, dtype=torch.long), graph


def collate_fn(batch):
    imgs   = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    graphs = [b[2] for b in batch if b[2] is not None]
    graph_batch = Batch.from_data_list(graphs) if graphs else None
    return imgs, labels, graph_batch

# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('  Hybrid Alignment — re-inference on baselines test set')
    print('=' * 70)
    print(f'  Device      : {DEVICE}')
    print(f'  Data dir    : {DATA_DIR}')
    print(f'  Model ckpt  : {MODEL_PATH}')
    print(f'  Graph cache : {GRAPH_CACHE_PATH}')

    for p in [DATA_DIR, MODEL_PATH, GRAPH_CACHE_PATH]:
        if not os.path.exists(p):
            sys.exit(f'\nERROR: path not found: {p}')

    # ── Load graph cache ─────────────────────────────────────────────
    print(f'\n[1/5] Loading Hybrid graph cache ...')
    with open(GRAPH_CACHE_PATH, 'rb') as f:
        graph_cache = pickle.load(f)
    print(f'      Loaded {len(graph_cache):,} cached graphs')

    # ── Rebuild the baselines test set (identical algorithm to LGCN) ─
    print(f'\n[2/5] Rebuilding baselines test set (LGCN/GAT/JGCN algorithm) ...')
    data_dir = Path(DATA_DIR)
    all_paths, all_labels = [], []
    for label_idx, cls in enumerate(['Benign', 'Malignant']):
        class_dir = data_dir / cls
        if not class_dir.exists():
            for alt in [cls.capitalize(), cls.upper()]:
                if (data_dir / alt).exists():
                    class_dir = data_dir / alt; break
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        all_paths.extend([str(p) for p in files])
        all_labels.extend([label_idx] * len(files))
        print(f'      {cls:12s}: {len(files):5d} images')
    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)
    print(f'      Total: {len(all_paths)} images')

    idx = np.arange(len(all_paths))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=all_labels, random_state=SEED)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50,
        stratify=all_labels[temp_idx], random_state=SEED)

    test_paths  = all_paths[test_idx]
    test_labels = all_labels[test_idx]
    print(f'      Test set: {len(test_paths)} images '
          f'({(test_labels == 0).sum()} benign, {(test_labels == 1).sum()} malignant)')

    # Cache-hit sanity check
    hits = sum(1 for p in test_paths if p in graph_cache)
    print(f'      Graph-cache hits: {hits} / {len(test_paths)}')
    if hits < len(test_paths):
        misses = [p for p in test_paths if p not in graph_cache][:3]
        print(f'      WARNING: {len(test_paths) - hits} paths not in graph cache. First 3 misses:')
        for m in misses: print(f'        {m}')
        print(f'      These samples will get graph=None (branch 3 outputs zeros).')

    # ── Build dataset + loader ───────────────────────────────────────
    print(f'\n[3/5] Building dataset + loader ...')
    eval_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    test_file_list = list(zip(test_paths.tolist(), test_labels.tolist()))
    test_ds = ProstateDataset(test_file_list, eval_tf, graph_cache)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    # ── Build model + load checkpoint ────────────────────────────────
    print(f'\n[4/5] Building model + loading checkpoint ...')
    model = HybridProstateCancerNet(
        num_classes=2, hidden=HIDDEN,
        gat_heads=GAT_HEADS, dropout=DROPOUT).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model', ckpt)   # supports both raw and dict-wrapped
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'      WARNING: {len(missing)} missing keys (first 3): {missing[:3]}')
    if unexpected:
        print(f'      WARNING: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}')
    print(f'      Loaded (best_epoch={ckpt.get("epoch", "?")}, '
          f'best_val_auc={ckpt.get("best_auc", "?")})')
    model.eval()

    # ── Run inference ────────────────────────────────────────────────
    print(f'\n[5/5] Running non-TTA inference on {len(test_ds)} samples ...')
    y_prob_all, y_pred_all, y_true_all = [], [], []
    paths_in_order = [test_file_list[i][0] for i in range(len(test_ds))]

    with torch.no_grad():
        for bi, (imgs, lbls, graphs) in enumerate(test_dl):
            imgs = imgs.to(DEVICE, non_blocking=True)
            if graphs is not None:
                graphs = graphs.to(DEVICE)
            out   = model(imgs, graphs)
            probs = torch.softmax(out.float(), dim=1)[:, 1]
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            preds = (probs >= 0.5).long()
            y_prob_all.extend(probs.cpu().numpy().tolist())
            y_pred_all.extend(preds.cpu().numpy().tolist())
            y_true_all.extend(lbls.numpy().tolist())
            if (bi + 1) % 5 == 0 or (bi + 1) == len(test_dl):
                print(f'      Batch {bi + 1:2d} / {len(test_dl)}')

    y_prob = np.array(y_prob_all)
    y_pred = np.array(y_pred_all)
    y_true = np.array(y_true_all)

    acc = (y_pred == y_true).mean()
    print(f'\n  Aligned Hybrid accuracy on baselines test set: {acc * 100:.2f}%')
    print(f'  (For reference, Hybrid original test set gave 98.34%)')

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        file_paths = np.array(paths_in_order),
        y_true     = y_true,
        y_pred     = y_pred,
        y_prob     = y_prob,
    )
    print(f'\n  Saved: {OUT_PATH}')
    print('\n  Next: python mcnemar_aligned.py')
    print('=' * 70)

if __name__ == '__main__':
    main()
