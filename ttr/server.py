#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
焊点火花检测 - 部署服务版 (多项目支持 + 多端口 + 长连接)

功能:
- 启动多个 TCP socket 服务器 (默认端口 5000-5009，共10个)
- 每个端口支持长连接，客户端可以连续发送多条命令
- 支持多项目，通过 projects.json 配置
- 模型按需加载并缓存，所有端口共享使用
- 另一个视觉软件每次拍完图片:
    1) 把图片保存到对应项目的 INPUT_DIR
    2) 通过 socket 发送 "RUN 项目ID\n" (如 "RUN P1\n")
- 本程序收到命令后:
    1) 根据项目ID加载对应的模型和目录配置
    2) 扫描 INPUT_DIR 下所有图片 (jpg/png/jpeg/bmp)
    3) 用训练好的模型分类 -> OK / NG
    4) 在图片上画一个 OK/NG 标记
    5) 保存到 OUTPUT_DIR
    6) 把原图移动到 DONE_DIR
    7) 返回 "OK <数量>\n"
- 客户端发送 "QUIT\n" 或断开连接时，服务器关闭该连接

命令格式:
- "RUN P1\n" - 使用项目P1的配置处理图片
- "RUN\n" - 使用默认项目处理图片
- "LIST\n" - 列出所有可用项目
- "QUIT\n" - 断开连接
"""

from pathlib import Path
import socket
import shutil
import threading

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont

from config_loader import get_project_config, get_default_project, list_projects, load_config


# ============ 默认配置 ============
IMG_SIZE = 224
HOST = "0.0.0.0"
PORT_START = 5000      # 起始端口
PORT_END   = 5009      # 结束端口 (包含)，共10个端口
USE_THRESHOLD = False
NG_THRESH = 0.5
# ================================

# 模型缓存: project_id -> (model, idx_ok, idx_ng, img_size, config)
_model_cache = {}
# 全局锁，保护模型推理和模型缓存（多线程共享时需要）
_model_lock = threading.Lock()


def infer_ok_ng_idx(class_to_idx: dict):
    """根据类名推断 OK / NG 的索引"""
    lower_map = {name.lower(): idx for name, idx in class_to_idx.items()}
    idx_ok = lower_map.get("ok", None)
    idx_ng = lower_map.get("ng", None)
    if idx_ok is None or idx_ng is None:
        raise ValueError(f"class_to_idx 中找不到 OK/NG 类，请检查: {class_to_idx}")
    return idx_ok, idx_ng


def build_model(num_classes: int):
    """构建与训练时相同结构的 ResNet18 分类模型"""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model_for_project(project_id: str, device):
    """加载指定项目的模型和类别信息，带缓存"""
    global _model_cache
    
    if project_id in _model_cache:
        print(f"使用缓存的模型: {project_id}")
        return _model_cache[project_id]
    
    config = get_project_config(project_id)
    model_path = Path(config.get("model_path", "D:/weld_data/weld_resnet18_cls.pth"))
    
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    print(f"加载项目 {project_id} 的模型: {model_path}")
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        class_to_idx = ckpt.get("class_to_idx", None)
        idx_ok = ckpt.get("idx_ok", None)
        idx_ng = ckpt.get("idx_ng", None)
        img_size = ckpt.get("img_size", IMG_SIZE)
    else:
        state_dict = ckpt
        class_to_idx = None
        idx_ok = idx_ng = None
        img_size = IMG_SIZE

    num_classes = state_dict[list(state_dict.keys())[-1]].shape[0]
    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if class_to_idx is not None:
        if idx_ok is None or idx_ng is None:
            idx_ok, idx_ng = infer_ok_ng_idx(class_to_idx)
    else:
        print("[警告] ckpt 中没有 class_to_idx，默认假设 0=NG,1=OK")
        idx_ng, idx_ok = 0, 1

    print(f"项目 {project_id} 类索引: idx_ok={idx_ok}, idx_ng={idx_ng}")
    
    _model_cache[project_id] = (model, idx_ok, idx_ng, img_size, config)
    return model, idx_ok, idx_ng, img_size, config


def ensure_dirs(config: dict):
    """确保项目的输入/输出目录存在"""
    input_dir = Path(config.get("input_dir", "D:/weld_runtime/in"))
    output_dir = Path(config.get("output_dir", "D:/weld_runtime/out"))
    done_dir = Path(config.get("done_dir", "D:/weld_runtime/done"))
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    
    return input_dir, output_dir, done_dir


def classify_and_mark_all(model, idx_ok, idx_ng, img_size, device, input_dir, output_dir, done_dir):
    """处理指定目录下的所有图片，返回处理数量"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    files = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    if not files:
        print("没有新图片需要处理。")
        return 0

    print(f"开始处理图片数量: {len(files)}")

    for img_path in files:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[错误] 打开图片失败 {img_path}: {e}")
            continue

        # 做推理（加锁保护，多线程共享模型）
        inp = tfm(img).unsqueeze(0).to(device)
        with _model_lock:
            with torch.no_grad():
                logits = model(inp)
                probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

        prob_ok = float(probs[idx_ok])
        prob_ng = float(probs[idx_ng])

        if USE_THRESHOLD:
            if prob_ng > NG_THRESH:
                label = "NG"
            else:
                label = "OK"
        else:
            pred_idx = int(torch.argmax(probs).item())
            label = "OK" if pred_idx == idx_ok else "NG"

        marked = draw_label_on_image(img, label, prob_ok, prob_ng)

        dst_out = output_dir / img_path.name
        marked.save(dst_out)

        # 输出同名的 txt 文件，包含推理结果和分数
        txt_name = img_path.stem + ".txt"
        txt_path = output_dir / txt_name
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"result={label}\n")
            f.write(f"prob_OK={prob_ok:.4f}\n")
            f.write(f"prob_NG={prob_ng:.4f}\n")

        dst_done = done_dir / img_path.name
        try:
            shutil.move(str(img_path), str(dst_done))
        except Exception as e:
            print(f"[警告] 移动原图失败 {img_path} -> {dst_done}: {e}")

        print(f"{img_path.name}: label={label}, prob_OK={prob_ok:.3f}, prob_NG={prob_ng:.3f}")

    return len(files)


def draw_label_on_image(img: Image.Image, label: str, prob_ok: float, prob_ng: float):
    """在图片左上角写上 OK/NG 标记及概率"""
    draw = ImageDraw.Draw(img)

    text = f"{label}  OK={prob_ok:.2f} NG={prob_ng:.2f}"

    # 选颜色: OK 绿色, NG 红色
    if label == "OK":
        fill = (0, 255, 0)
    else:
        fill = (255, 0, 0)

    # 字体: 默认 PIL 内置字体，避免依赖 ttf
    font = ImageFont.load_default()

    # 用 textbbox 计算文本边界 (Pillow 10+ 推荐)
    # bbox = (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    margin = 4
    x0, y0 = 5, 5
    x1, y1 = x0 + text_w + 2 * margin, y0 + text_h + 2 * margin

    # 画底框 (黑底)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    # 再画文字
    draw.text((x0 + margin, y0 + margin), text, fill=fill, font=font)

    return img



def handle_client(conn, addr, port, device):
    """处理单个客户端连接（长连接，循环接收命令）"""
    print(f"[端口 {port}] 收到连接来自: {addr}")
    buffer = ""

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                # 客户端断开连接
                print(f"[端口 {port}] 客户端 {addr} 断开连接")
                break

            buffer += data.decode("utf-8", errors="ignore")

            # 按行处理命令（支持一次发送多条命令）
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                raw_cmd = line.strip()

                if not raw_cmd:
                    continue

                print(f"[端口 {port}] 收到命令: {raw_cmd}")

                parts = raw_cmd.split()
                cmd = parts[0].upper() if parts else ""

                if cmd == "QUIT":
                    conn.sendall(b"BYE\n")
                    print(f"[端口 {port}] 客户端 {addr} 请求退出")
                    return

                if cmd == "LIST":
                    projects = list_projects()
                    resp = f"OK {','.join(projects)}\n"

                elif cmd in ("RUN", "TRIGGER"):
                    project_id = parts[1] if len(parts) > 1 else get_default_project()

                    try:
                        with _model_lock:
                            model, idx_ok, idx_ng, img_size, config = load_model_for_project(project_id, device)
                        input_dir, output_dir, done_dir = ensure_dirs(config)

                        print(f"[端口 {port}] 处理项目 {project_id}: input={input_dir}, output={output_dir}")
                        count = classify_and_mark_all(
                            model, idx_ok, idx_ng, img_size, device,
                            input_dir, output_dir, done_dir
                        )
                        resp = f"OK {count}\n"
                    except ValueError as e:
                        resp = f"ERR UNKNOWN_PROJECT {project_id}\n"
                        print(f"[端口 {port}] [错误] 未知项目: {e}")
                    except FileNotFoundError as e:
                        resp = f"ERR MODEL_NOT_FOUND\n"
                        print(f"[端口 {port}] [错误] 模型文件不存在: {e}")

                else:
                    resp = "ERR UNKNOWN_CMD\n"

                conn.sendall(resp.encode("utf-8"))
                print(f"[端口 {port}] 已回复: {resp.strip()}")

    except Exception as e:
        print(f"[端口 {port}] [错误] 处理连接出错: {e}")
        try:
            conn.sendall(b"ERR SERVER_EXCEPTION\n")
        except Exception:
            pass
    finally:
        conn.close()


def port_listener(port, device):
    """单个端口的监听线程"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, port))
        s.listen(5)
        print(f"端口 {port} 已启动监听")

        while True:
            conn, addr = s.accept()
            # 为每个连接创建一个处理线程
            client_thread = threading.Thread(
                target=handle_client,
                args=(conn, addr, port, device),
                daemon=True
            )
            client_thread.start()


def run_server():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print("可用项目:", list_projects())
    print("默认项目:", get_default_project())

    print(f"\n启动多端口服务器，端口范围: {PORT_START}-{PORT_END}")
    print("每个端口支持长连接，客户端可以连续发送命令")
    print("命令格式: RUN [项目ID] 或 LIST 或 QUIT")
    print("发送 QUIT 命令或断开连接可关闭连接\n")

    threads = []
    for port in range(PORT_START, PORT_END + 1):
        t = threading.Thread(
            target=port_listener,
            args=(port, device),
            daemon=True
        )
        t.start()
        threads.append(t)

    # 主线程等待（保持程序运行）
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == "__main__":
    run_server()
