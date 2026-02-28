#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from PIL import Image

# ========= 你可在这里改默认路径 =========
DEFAULT_IMG_DIR = r"D:\PIC\LABELOK"
DEFAULT_JSON_DIR = r"D:\PIC\JSON"
DEFAULT_OUT_DIR = r"D:\PIC\glyph_bank"

# Windows 文件夹名不能用这些字符：<>:"/\|?*
SPECIAL_MAP = {
    "/": "slash",
    "\\": "backslash",
    ".": "dot",
    ":": "colon",
    "*": "asterisk",
    "?": "question",
    "\"": "quote",
    "<": "lt",
    ">": "gt",
    "|": "pipe",
    " ": "space",
    "\t": "tab",
}

INVALID_WIN = set('<>:"/\\|?*')


def safe_folder_name(ch: str) -> str:
    if ch in SPECIAL_MAP:
        return SPECIAL_MAP[ch]
    if ch in INVALID_WIN:
        return f"u{ord(ch):04X}"
    # 避免末尾点/空格
    if ch.endswith(".") or ch.endswith(" "):
        return f"u{ord(ch):04X}"
    return ch


def clip_box(x0: int, y0: int, x1: int, y1: int, W: int, H: int):
    x0 = max(0, min(x0, W))
    y0 = max(0, min(y0, H))
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return x0, y0, x1, y1


def next_index_file(folder: Path, ext=".jpg") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for fp in folder.glob(f"*{ext}"):
        stem = fp.stem
        if stem.isdigit():
            n = int(stem)
            if n > max_n:
                max_n = n
    return folder / f"{max_n + 1:04d}{ext}"


def crop_one_json(json_path: Path, img_dir: Path, out_dir: Path, pad: int = 2, ext=".jpg", verbose=True):
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))

    image_name = data.get("image_name", "")
    if not image_name:
        if verbose:
            print(f"[SKIP] no image_name in {json_path.name}")
        return 0

    img_path = img_dir / image_name
    if not img_path.exists():
        # 兼容：万一 json 里带了路径，只取文件名
        img_path = img_dir / Path(image_name).name

    if not img_path.exists():
        if verbose:
            print(f"[MISS] image not found for {json_path.name}: {img_path}")
        return 0

    items = data.get("items", [])
    if not items:
        if verbose:
            print(f"[SKIP] no items: {json_path.name}")
        return 0

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    saved = 0
    for it in items:
        ch = str(it.get("ch", ""))
        if ch == "":
            continue

        cx = float(it.get("cx", 0))
        cy = float(it.get("cy", 0))
        w = float(it.get("w", 0))
        h = float(it.get("h", 0))
        if w <= 0 or h <= 0:
            continue

        x0 = int(round(cx - w / 2.0)) - pad
        y0 = int(round(cy - h / 2.0)) - pad
        x1 = int(round(cx + w / 2.0)) + pad
        y1 = int(round(cy + h / 2.0)) + pad
        x0, y0, x1, y1 = clip_box(x0, y0, x1, y1, W, H)

        crop = img.crop((x0, y0, x1, y1))

        folder = out_dir / safe_folder_name(ch)
        fp = next_index_file(folder, ext=ext)
        crop.save(fp, quality=95)
        saved += 1

    if verbose:
        print(f"[OK] {json_path.name} -> {img_path.name}  saved={saved}")
    return saved


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=DEFAULT_IMG_DIR, help="images directory, e.g. D:\\PIC\\LABELOK")
    ap.add_argument("--json_dir", default=DEFAULT_JSON_DIR, help="json directory, e.g. D:\\PIC\\JSON")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="output glyph bank directory")
    ap.add_argument("--pad", type=int, default=2, help="padding pixels around each box")
    ap.add_argument("--ext", default=".jpg", help="output image extension, default .jpg")
    ap.add_argument("--quiet", action="store_true", help="less logs")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"[ERR] no json found in {json_dir}")
        return

    total = 0
    for jp in json_files:
        total += crop_one_json(jp, img_dir, out_dir, pad=args.pad, ext=args.ext, verbose=not args.quiet)

    print(f"[DONE] total saved crops = {total}")
    print(f"[DONE] output dir = {out_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from PIL import Image

# ========= 你可在这里改默认路径 =========
DEFAULT_IMG_DIR = r"D:\PIC\LABELOK"
DEFAULT_JSON_DIR = r"D:\PIC\JSON"
DEFAULT_OUT_DIR = r"D:\glyph_bank"

# Windows 文件夹名不能用这些字符：<>:"/\|?*
SPECIAL_MAP = {
    "/": "slash",
    "\\": "backslash",
    ".": "dot",
    ":": "colon",
    "*": "asterisk",
    "?": "question",
    "\"": "quote",
    "<": "lt",
    ">": "gt",
    "|": "pipe",
    " ": "space",
    "\t": "tab",
}

INVALID_WIN = set('<>:"/\\|?*')


def safe_folder_name(ch: str) -> str:
    if ch in SPECIAL_MAP:
        return SPECIAL_MAP[ch]
    if ch in INVALID_WIN:
        return f"u{ord(ch):04X}"
    # 避免末尾点/空格
    if ch.endswith(".") or ch.endswith(" "):
        return f"u{ord(ch):04X}"
    return ch


def clip_box(x0: int, y0: int, x1: int, y1: int, W: int, H: int):
    x0 = max(0, min(x0, W))
    y0 = max(0, min(y0, H))
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return x0, y0, x1, y1


def next_index_file(folder: Path, ext=".jpg") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for fp in folder.glob(f"*{ext}"):
        stem = fp.stem
        if stem.isdigit():
            n = int(stem)
            if n > max_n:
                max_n = n
    return folder / f"{max_n + 1:04d}{ext}"


def crop_one_json(json_path: Path, img_dir: Path, out_dir: Path, pad: int = 2, ext=".jpg", verbose=True):
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))

    image_name = data.get("image_name", "")
    if not image_name:
        if verbose:
            print(f"[SKIP] no image_name in {json_path.name}")
        return 0

    img_path = img_dir / image_name
    if not img_path.exists():
        # 兼容：万一 json 里带了路径，只取文件名
        img_path = img_dir / Path(image_name).name

    if not img_path.exists():
        if verbose:
            print(f"[MISS] image not found for {json_path.name}: {img_path}")
        return 0

    items = data.get("items", [])
    if not items:
        if verbose:
            print(f"[SKIP] no items: {json_path.name}")
        return 0

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    saved = 0
    for it in items:
        ch = str(it.get("ch", ""))
        if ch == "":
            continue

        cx = float(it.get("cx", 0))
        cy = float(it.get("cy", 0))
        w = float(it.get("w", 0))
        h = float(it.get("h", 0))
        if w <= 0 or h <= 0:
            continue

        x0 = int(round(cx - w / 2.0)) - pad
        y0 = int(round(cy - h / 2.0)) - pad
        x1 = int(round(cx + w / 2.0)) + pad
        y1 = int(round(cy + h / 2.0)) + pad
        x0, y0, x1, y1 = clip_box(x0, y0, x1, y1, W, H)

        crop = img.crop((x0, y0, x1, y1))

        folder = out_dir / safe_folder_name(ch)
        fp = next_index_file(folder, ext=ext)
        crop.save(fp, quality=95)
        saved += 1

    if verbose:
        print(f"[OK] {json_path.name} -> {img_path.name}  saved={saved}")
    return saved


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=DEFAULT_IMG_DIR, help="images directory, e.g. D:\\PIC\\LABELOK")
    ap.add_argument("--json_dir", default=DEFAULT_JSON_DIR, help="json directory, e.g. D:\\PIC\\JSON")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="output glyph bank directory")
    ap.add_argument("--pad", type=int, default=2, help="padding pixels around each box")
    ap.add_argument("--ext", default=".jpg", help="output image extension, default .jpg")
    ap.add_argument("--quiet", action="store_true", help="less logs")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"[ERR] no json found in {json_dir}")
        return

    total = 0
    for jp in json_files:
        total += crop_one_json(jp, img_dir, out_dir, pad=args.pad, ext=args.ext, verbose=not args.quiet)

    print(f"[DONE] total saved crops = {total}")
    print(f"[DONE] output dir = {out_dir}")


if __name__ == "__main__":
    main()
