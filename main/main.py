# ===== 1️⃣ IMPORTS =====
import httpx
import torch
import io
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import os
# ===== 2️⃣ ĐỌC ẢNH TEST =====
try:
    with open("test.jpg", "rb") as f:
        image_bytes = f.read()
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file test.jpg")
    exit()

# ===== 3️⃣ GỬI SANG ENCODER =====
try:
    with httpx.Client() as client:
        enc_res = client.post(
            "http://127.0.0.1:8001/encode",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            timeout=30.0
        )
        enc_res.raise_for_status()
        feat_binary = enc_res.content
    print("✅ Encoder OK")
except Exception as e:
    print(f"❌ Lỗi Encoder: {e}")
    exit()

# ===== 4️⃣ GỬI SANG DECODER =====
try:
    with httpx.Client() as client:
        dec_res = client.post(
            "http://127.0.0.1:8002/decode",
            content=feat_binary,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30.0
        )
        dec_res.raise_for_status()
        output_binary = dec_res.content
    print("✅ Decoder OK")
except Exception as e:
    print(f"❌ Lỗi Decoder: {e}")
    exit()

# ===== 5️⃣ LOAD TENSOR OUTPUT =====
try:
    buffer = io.BytesIO(output_binary)
    output = torch.load(buffer, map_location="cpu", weights_only=False)
    print(f"📊 Type: {type(output)}")
    
    if isinstance(output, dict):
        print(f"📊 Keys: {list(output.keys())}")
        # Debug: In shape của các tensor trong dict
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}.shape: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
    elif isinstance(output, torch.Tensor):
        print(f"📊 Shape: {output.shape}")
    else:
        print(f"📊 Output: {output}")
except Exception as e:
    print(f"❌ Lỗi load tensor: {e}")
    exit()

# ===== 6️⃣ VẼ DETECTION =====
try:
    # Kiểm tra keys cần thiết
    required_keys = ["pred_boxes", "pred_logits"]
    for key in required_keys:
        if key not in output:
            print(f"❌ Lỗi: Key '{key}' không tồn tại trong output")
            print(f"Available keys: {list(output.keys())}")
            exit()
    
    boxes = output["pred_boxes"][0]      # [num_queries, 4]
    logits = output["pred_logits"][0]    # [num_queries, num_classes]
    
    print(f"📊 Boxes shape: {boxes.shape}")
    print(f"📊 Logits shape: {logits.shape}")
    
    # Tính xác suất
    probs = F.softmax(logits, dim=-1)
    
    # Bỏ class cuối nếu là background (thường DETR có)
    scores, labels = probs[..., :-1].max(dim=-1)
    
    # Lọc confidence
    threshold = 0.7
    keep = scores > threshold
    
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    print(f"📊 Found {len(boxes)} objects above threshold {threshold}")
    
    if len(boxes) == 0:
        print("⚠️ Không có object nào vượt threshold.")
        # Vẫn vẽ ảnh gốc để debug
        image = Image.open("test.jpg")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title("No Objects Detected")
        plt.axis("off")
        plt.savefig("result.jpg", bbox_inches="tight", pad_inches=0)
        plt.close()
        exit()
    
    # Load ảnh gốc
    image = Image.open("test.jpg")
    w, h = image.size
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # Kiểm tra box format
    print(f"📊 Sample box: {boxes[0].tolist()}")
    
    for box, score, label in zip(boxes, scores, labels):
        # D-FINE/DETR thường dùng format: [cx, cy, w, h] (normalized 0-1)
        cx, cy, bw, bh = box.tolist()
        
        # Convert từ center format sang xyxy
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
        
        # Clamp để không vượt ảnh
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Thêm label
        ax.text(x1, y1 - 5, f"Class {label.item()}: {score.item():.2f}", 
                color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis("off")
    
    # ===== LƯU ẢNH =====
    output_path = "result.jpg"
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    print(f"✅ Đã lưu kết quả vào {output_path}")
    print("Current working directory:", os.getcwd())
except Exception as e:
    print(f"❌ Lỗi vẽ detection: {e}")
    import traceback
    traceback.print_exc()