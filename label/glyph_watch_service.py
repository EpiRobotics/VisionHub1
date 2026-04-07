#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import time, json, shutil, threading, traceback
from dataclasses import dataclass
from queue import Queue
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import cv2
import joblib
import torch
import torch.nn as nn
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ===================== 现场只改这里 =====================
IN_DIR = Path(r"D:\in\site_incoming")
OUT_OVERLAY_DIR = Path(r"D:\in\overlay")
ARCHIVE_DIR = Path(r"D:\in\site_archive")
ERROR_DIR = Path(r"D:\in\site_error")

MODEL_DIR = Path(r"D:\models_patchcore")   # 你压缩后的目录

PAD = 1
THR_GLOBAL: Optional[float] = 2.05

USE_GPU = True
USE_FP16 = True

CNN_BATCH = 64

KNN_ON_GPU = True

# --- Small-defect enhancement options ---
# "layer2" = original (128-dim, 8x8 patches)
# "multi"  = layer1+layer2 concatenated (192-dim, 16x16 patches, 4x more patches)
#            Requires models trained with feature_layers="multi"
FEATURE_LAYERS = "layer2"
# CLAHE local contrast enhancement (0 = disabled, try 1.0-3.0 for small defects)
# Requires models trained with matching clahe_clip value
CLAHE_CLIP = 0.0
# bank 已经每类~15000，block 直接开大：减少循环次数
KNN_BANK_BLOCK = 20000

STABLE_INTERVAL_SEC = 0.01
STABLE_CONSECUTIVE_N = 2
STABLE_MAX_WAIT_SEC = 6.0

PAIR_TIMEOUT_SEC = 12.0
PERIODIC_SCAN_SEC = 1.0

MAX_RETRY = 2
RETRY_DELAY_SEC = 0.4

OUT_EXT = ".png"
PRINT_PER_GLYPH = True
# =======================================================

SPECIAL_MAP = {
    "/": "slash", "\\": "backslash", ".": "dot", ":": "colon",
    "*": "asterisk", "?": "question", "\"": "quote",
    "<": "lt", ">": "gt", "|": "pipe", " ": "space", "\t": "tab",
}
INVALID_WIN = set('<>:"/\\|?*')
IMG_EXT = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def safe_folder_name(ch: str) -> str:
    if ch in SPECIAL_MAP: return SPECIAL_MAP[ch]
    if ch in INVALID_WIN: return f"u{ord(ch):04X}"
    if ch.endswith(".") or ch.endswith(" "): return f"u{ord(ch):04X}"
    return ch

def imread_any_bgr(p: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0: return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_any(p: Path, bgr: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix.lower() if p.suffix else ".png", bgr)
    if not ok: raise RuntimeError("cv2.imencode failed")
    buf.tofile(str(p))

def clip_box(x0,y0,x1,y1,W,H):
    x0=max(0,min(int(x0),W)); y0=max(0,min(int(y0),H))
    x1=max(0,min(int(x1),W)); y1=max(0,min(int(y1),H))
    if x1<=x0: x1=min(W,x0+1)
    if y1<=y0: y1=min(H,y0+1)
    return x0,y0,x1,y1

def wait_file_stable(p: Path,
                     interval=STABLE_INTERVAL_SEC,
                     consecutive=STABLE_CONSECUTIVE_N,
                     max_wait=STABLE_MAX_WAIT_SEC) -> bool:
    last = -1
    same = 0
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < max_wait:
        if not p.exists():
            time.sleep(interval); continue
        try:
            sz = p.stat().st_size
        except OSError:
            time.sleep(interval); continue
        if sz > 0 and sz == last:
            same += 1
            if same >= consecutive:
                return True
        else:
            same = 0
            last = sz
        time.sleep(interval)
    return False

def cuda_sync_if_needed(device: str):
    if device == "cuda":
        torch.cuda.synchronize()

def move_to_dir(p: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / p.name
    if dst.exists():
        ts = time.strftime("%H%M%S")
        dst = dst_dir / f"{p.stem}_{ts}{p.suffix}"
    try:
        p.replace(dst)
    except Exception:
        shutil.copy2(str(p), str(dst))
        try: p.unlink(missing_ok=True)
        except Exception: pass

def archive_pair(img_path: Path, json_path: Path):
    day_dir = ARCHIVE_DIR / time.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    move_to_dir(img_path, day_dir)
    move_to_dir(json_path, day_dir)

# ------------------ model ------------------
class ResNetFeat(nn.Module):
    def __init__(self, feature_layers: str = "layer2"):
        super().__init__()
        from torchvision import models
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.feature_layers = feature_layers
    def forward(self, x):
        if self.feature_layers == "multi":
            return self._forward_multiscale(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    def _forward_multiscale(self, x):
        """Multi-scale: concat layer1 (64-dim, 16x16) + upsampled layer2 (128-dim).
        Result: [B, 192, H1, W1] with 4x more patches for small-defect sensitivity."""
        x = self.stem(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat2_up = nn.functional.interpolate(
            feat2, size=feat1.shape[2:], mode="bilinear", align_corners=False,
        )
        return torch.cat([feat1, feat2_up], dim=1)

_TF = None
def _get_tf():
    global _TF
    if _TF is None:
        from torchvision import transforms
        _TF = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
    return _TF

def preprocess_gray_to_tensor(gray: np.ndarray, size: int,
                               clahe_clip: float = 0.0) -> torch.Tensor:
    g = cv2.resize(gray, (size,size), interpolation=cv2.INTER_CUBIC)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
        g = clahe.apply(g)
    rgb = np.stack([g,g,g], axis=-1).astype(np.uint8)
    return _get_tf()(rgb).unsqueeze(0)  # [1,3,S,S]

def extract_patch_embeddings_batch(fm: torch.Tensor) -> torch.Tensor:
    # fm: [B,C,H,W] -> [B, N, C]
    B,C,H,W = fm.shape
    return fm.permute(0,2,3,1).reshape(B, H*W, C)

def knn_mean_distance_torch(
    emb_t: torch.Tensor,      # [M,C] on cuda/float16
    bank_t: torch.Tensor,     # [B,C] on cuda/float16
    bank_norm: torch.Tensor,  # [B] cuda/float32
    k: int,
    block: int
) -> torch.Tensor:
    emb_norm = (emb_t.to(torch.float32) * emb_t.to(torch.float32)).sum(dim=1)  # [M]
    M = emb_t.shape[0]
    k = max(1, int(k))
    best = torch.full((M, k), float("inf"), device=emb_t.device, dtype=torch.float32)

    B = bank_t.shape[0]
    for s in range(0, B, block):
        e = min(B, s + block)
        bt = bank_t[s:e]                       # [b,C] fp16
        bn = bank_norm[s:e]                    # [b] float32
        dot = emb_t.to(torch.float32) @ bt.to(torch.float32).T  # [M,b]
        dist2 = emb_norm[:, None] + bn[None, :] - 2.0 * dot
        dist2 = torch.clamp(dist2, min=0.0)

        kk = min(k, dist2.shape[1])
        blk_best, _ = torch.topk(dist2, k=kk, largest=False, dim=1)
        cat = torch.cat([best, blk_best], dim=1)
        best, _ = torch.topk(cat, k=k, largest=False, dim=1)

    return torch.sqrt(best).mean(dim=1)  # [M]

def score_from_patch_d(d_patch: torch.Tensor, score_mode: str, topk: int) -> float:
    # d_patch: [N] torch tensor on CPU/GPU
    if score_mode == "max":
        return float(d_patch.max().item())
    topk = max(1, min(int(topk), d_patch.numel()))
    v = torch.topk(d_patch, k=topk, largest=True).values
    topk_mean = float(v.mean().item())
    if score_mode == "adaptive":
        max_val = float(d_patch.max().item())
        return float(np.sqrt(max_val * topk_mean))
    return topk_mean

@dataclass
class ClassModel:
    cls: str
    img_size: int
    k: int
    score_mode: str
    topk: int
    thr: float
    bank_t: torch.Tensor
    bank_norm: torch.Tensor
    feature_layers: str = "layer2"
    clahe_clip: float = 0.0

class PatchCoreSingle:
    def __init__(self, model_dir: Path):
        self.device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
        self.fp16 = bool(USE_FP16 and self.device == "cuda")
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.cls_models: Dict[str, ClassModel] = {}

        files = sorted(Path(model_dir).glob("*.joblib"))
        if not files:
            raise RuntimeError(f"no *.joblib found in {model_dir}")

        total_bank = 0
        for fp in files:
            m = joblib.load(fp)
            cls = str(m["cls"])
            bank = np.asarray(m["memory_bank"], dtype=np.float32)  # [B,C]
            total_bank += bank.shape[0]

            img_size = int(m.get("img_size", 128))
            k = int(m.get("k", 1))
            score_mode = str(m.get("score_mode", "topk"))
            topk = int(m.get("topk", 10))
            thr = float(m.get("thr", 1e9))
            model_fl = str(m.get("feature_layers", FEATURE_LAYERS))
            model_cc = float(m.get("clahe_clip", CLAHE_CLIP))

            use_knn_gpu = (KNN_ON_GPU and self.device == "cuda")
            dev = torch.device("cuda") if use_knn_gpu else torch.device("cpu")
            bank_t = torch.from_numpy(bank).to(dev, dtype=torch.float16 if use_knn_gpu else torch.float32)
            bank_norm = (bank_t.to(torch.float32) * bank_t.to(torch.float32)).sum(dim=1)

            self.cls_models[cls] = ClassModel(cls, img_size, k, score_mode, topk, thr, bank_t, bank_norm,
                                              feature_layers=model_fl, clahe_clip=model_cc)

        # Detect feature_layers from loaded models
        effective_fl = FEATURE_LAYERS
        if self.cls_models:
            effective_fl = next(iter(self.cls_models.values())).feature_layers
        self.net = ResNetFeat(feature_layers=effective_fl).to(self.device).eval()

        print(f"[MODEL] loaded classes: {len(self.cls_models)} from {model_dir}  total_bank={total_bank}")
        print(f"[MODEL] device={self.device}, fp16={self.fp16}, knn_on_gpu={KNN_ON_GPU and self.device=='cuda'}, feature_layers={effective_fl}")
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((64,64), dtype=np.uint8)
        x = preprocess_gray_to_tensor(dummy, 128).to(self.device)
        with torch.no_grad():
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = self.net(x)
            else:
                _ = self.net(x)
        cuda_sync_if_needed(self.device)
        print("[MODEL] warmup done")

    def predict_one(self, img_path: Path, json_path: Path, out_overlay: Path,
                    pad: int = 1, thr_global: Optional[float] = None) -> Tuple[int,int,float,float,Dict[str,Any]]:
        bgr = imread_any_bgr(img_path)
        if bgr is None:
            raise RuntimeError(f"imread failed: {img_path}")
        H, W = bgr.shape[:2]
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []

        vis = bgr.copy()
        ng_count = 0
        unk_count = 0

        glyphs = []
        for it in items:
            ch = str(it.get("ch",""))
            if not ch: continue
            cx = float(it.get("cx",0)); cy = float(it.get("cy",0))
            ww = float(it.get("w",0));  hh = float(it.get("h",0))
            if ww<=0 or hh<=0: continue

            x0 = int(round(cx-ww/2)) - pad
            y0 = int(round(cy-hh/2)) - pad
            x1 = int(round(cx+ww/2)) + pad
            y1 = int(round(cy+hh/2)) + pad
            x0,y0,x1,y1 = clip_box(x0,y0,x1,y1,W,H)

            cls = safe_folder_name(ch)
            if cls not in self.cls_models:
                unk_count += 1
                cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)
                continue

            cm = self.cls_models[cls]
            thr = float(thr_global) if thr_global is not None else cm.thr
            patch = gray_full[y0:y1, x0:x1]

            glyphs.append({"ch":ch,"cls":cls,"cm":cm,"thr":thr,"box":(x0,y0,x1,y1),"patch":patch,"emb_t":None})

        if not glyphs:
            imwrite_any(out_overlay, vis)
            return 0, unk_count, 0.0, 0.0, {"glyph_total": 0}

        # ---- CNN ----
        cuda_sync_if_needed(self.device)
        t_cnn0 = time.perf_counter()

        groups: Dict[int, List[int]] = {}
        for i,g in enumerate(glyphs):
            groups.setdefault(int(g["cm"].img_size), []).append(i)

        with torch.no_grad():
            for sz, idxs in groups.items():
                for s in range(0, len(idxs), CNN_BATCH):
                    batch_ids = idxs[s:s+CNN_BATCH]
                    xs = [preprocess_gray_to_tensor(glyphs[gi]["patch"], sz,
                                                     clahe_clip=glyphs[gi]["cm"].clahe_clip)
                          for gi in batch_ids]
                    x = torch.cat(xs, dim=0).to(self.device, non_blocking=False)

                    if self.fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            fm = self.net(x)
                        emb = extract_patch_embeddings_batch(fm).float()   # [B,N,C] float32 on GPU
                    else:
                        fm = self.net(x)
                        emb = extract_patch_embeddings_batch(fm)           # float32 on GPU

                    # 关键：不再 cpu().numpy()，直接把每个 glyph 的 [N,C] 留在 GPU
                    for bi, gi in enumerate(batch_ids):
                        glyphs[gi]["emb_t"] = emb[bi].contiguous()         # [N,C] on GPU

        cuda_sync_if_needed(self.device)
        t_cnn1 = time.perf_counter()
        cnn_ms = (t_cnn1 - t_cnn0) * 1000.0

        # ---- kNN (GPU, no roundtrip) ----
        t_knn0 = time.perf_counter()

        cls_map: Dict[str, List[int]] = {}
        for i,g in enumerate(glyphs):
            cls_map.setdefault(g["cls"], []).append(i)

        for cls, idxs in cls_map.items():
            cm: ClassModel = glyphs[idxs[0]]["cm"]
            # 拼成 [M,C]，M=len(idxs)*N
            emb_list = [glyphs[i]["emb_t"].reshape(-1, glyphs[i]["emb_t"].shape[-1]) for i in idxs]
            big_t = torch.cat(emb_list, dim=0).to("cuda", dtype=torch.float16)  # [M,C] fp16

            d = knn_mean_distance_torch(
                emb_t=big_t,
                bank_t=cm.bank_t,
                bank_norm=cm.bank_norm,
                k=cm.k,
                block=KNN_BANK_BLOCK
            )  # [M] cuda float32

            # split back
            N = glyphs[idxs[0]]["emb_t"].shape[0]
            for k_i, gi in enumerate(idxs):
                dd = d[k_i*N:(k_i+1)*N]
                score = score_from_patch_d(dd, cm.score_mode, cm.topk)
                thr = float(glyphs[gi]["thr"])
                is_ng = score > thr

                x0,y0,x1,y1 = glyphs[gi]["box"]
                ch = glyphs[gi]["ch"]
                if is_ng:
                    ng_count += 1
                    cv2.rectangle(vis, (x0,y0), (x1,y1), (0,0,255), 2)
                    cv2.putText(vis, f"{ch}:{score:.2f}", (x0, max(15,y0-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
                else:
                    cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 1)

                if PRINT_PER_GLYPH:
                    print(f"  {ch} score={score:.3f} thr={thr:.3f} is_ng={is_ng}")

        cuda_sync_if_needed(self.device)
        t_knn1 = time.perf_counter()
        knn_ms = (t_knn1 - t_knn0) * 1000.0

        imwrite_any(out_overlay, vis)
        return ng_count, unk_count, cnn_ms, knn_ms, {"glyph_total": len(glyphs)}

# ------------------ pairing/watch ------------------
@dataclass
class PendingFile:
    path: Path
    t_first_seen: float

class PairManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending_imgs: Dict[str, PendingFile] = {}
        self.pending_jsons: Dict[str, PendingFile] = {}

    def add_path(self, p: Path):
        ext = p.suffix.lower()
        if ext == ".json":
            with self.lock:
                self.pending_jsons[p.stem] = PendingFile(p, time.time())
        elif ext in IMG_EXT:
            with self.lock:
                self.pending_imgs[p.stem] = PendingFile(p, time.time())

    def pop_pair_if_ready(self) -> Optional[Tuple[Path, Path]]:
        with self.lock:
            for stem in list(self.pending_jsons.keys()):
                if stem in self.pending_imgs:
                    jp = self.pending_jsons.pop(stem).path
                    ip = self.pending_imgs.pop(stem).path
                    return ip, jp
        return None

    def collect_timeouts(self, timeout_sec: float) -> List[Path]:
        dead = []
        now = time.time()
        with self.lock:
            for dct in (self.pending_imgs, self.pending_jsons):
                for k in list(dct.keys()):
                    pf = dct[k]
                    if (now - pf.t_first_seen) >= timeout_sec:
                        dead.append(pf.path)
                        dct.pop(k, None)
        return dead

class WatchHandler(FileSystemEventHandler):
    def __init__(self, pm: PairManager):
        super().__init__()
        self.pm = pm
    def on_created(self, event):
        if not event.is_directory:
            self.pm.add_path(Path(event.src_path))
    def on_moved(self, event):
        if not event.is_directory:
            self.pm.add_path(Path(event.dest_path))

def scan_dir_once(pm: PairManager, in_dir: Path):
    try:
        for p in in_dir.iterdir():
            if not p.is_file(): continue
            ext = p.suffix.lower()
            if ext == ".json" or ext in IMG_EXT:
                pm.add_path(p)
    except Exception:
        pass

def worker_loop(q: Queue, predictor: PatchCoreSingle):
    while True:
        img_path, json_path, retry = q.get()
        t0 = time.perf_counter()
        try:
            if not img_path.exists() or not json_path.exists():
                continue

            t_wait0 = time.perf_counter()
            ok_img = wait_file_stable(img_path)
            ok_json = wait_file_stable(json_path)
            t_wait1 = time.perf_counter()
            wait_ms = (t_wait1 - t_wait0) * 1000.0

            if not ok_img or not ok_json:
                if retry < MAX_RETRY:
                    time.sleep(RETRY_DELAY_SEC)
                    q.put((img_path, json_path, retry+1))
                    print(f"[RETRY] {img_path.stem} retry={retry+1}/{MAX_RETRY} wait={wait_ms:.1f}ms")
                    continue
                raise RuntimeError("file not stable")

            stem = img_path.stem
            out_overlay = OUT_OVERLAY_DIR / f"{stem}_overlay{OUT_EXT}"

            ng, unk, cnn_ms, knn_ms, meta = predictor.predict_one(
                img_path, json_path, out_overlay, pad=PAD, thr_global=THR_GLOBAL
            )

            t_post0 = time.perf_counter()
            archive_pair(img_path, json_path)
            t_post1 = time.perf_counter()
            post_ms = (t_post1 - t_post0) * 1000.0

            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0

            print(f"[OK] {stem} NG={ng} UNK={unk} glyph={meta['glyph_total']} "
                  f"wait={wait_ms:.1f}ms cnn={cnn_ms:.1f}ms knn={knn_ms:.1f}ms post={post_ms:.1f}ms total={total_ms:.1f}ms  out={out_overlay.name}")

        except Exception as e:
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0
            print(f"[ERROR] {img_path.name} + {json_path.name} total={total_ms:.1f}ms err={e}")
            traceback.print_exc()
            try:
                if img_path.exists(): move_to_dir(img_path, ERROR_DIR)
                if json_path.exists(): move_to_dir(json_path, ERROR_DIR)
            except Exception:
                pass
        finally:
            q.task_done()

def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    predictor = PatchCoreSingle(MODEL_DIR)

    pm = PairManager()
    q: Queue = Queue()

    th = threading.Thread(target=worker_loop, args=(q, predictor), daemon=True)
    th.start()

    handler = WatchHandler(pm)
    obs = Observer()
    obs.schedule(handler, str(IN_DIR), recursive=False)
    obs.start()

    print("[SERVICE] watching:", IN_DIR)
    print("[SERVICE] model:", MODEL_DIR)
    print(f"[SERVICE] thr_global={THR_GLOBAL} pad={PAD} device={predictor.device} fp16={predictor.fp16} cnn_batch={CNN_BATCH} knn_block={KNN_BANK_BLOCK}")

    scan_dir_once(pm, IN_DIR)

    try:
        last_scan = time.time()
        while True:
            while True:
                pair = pm.pop_pair_if_ready()
                if not pair: break
                ip, jp = pair
                q.put((ip, jp, 0))

            dead = pm.collect_timeouts(PAIR_TIMEOUT_SEC)
            for p in dead:
                print(f"[TIMEOUT] move to error: {p.name}")
                if p.exists(): move_to_dir(p, ERROR_DIR)

            if (time.time() - last_scan) >= PERIODIC_SCAN_SEC:
                scan_dir_once(pm, IN_DIR)
                last_scan = time.time()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[SERVICE] stopping...")
        obs.stop()
    finally:
        obs.join()

if __name__ == "__main__":
    main()
