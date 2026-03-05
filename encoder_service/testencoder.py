import torch
import io
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from src.core.yaml_config import YAMLConfig

app = FastAPI()
cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model

# Load trọng số
backbone = model.backbone
encoder = model.encoder

# load weight riêng
backbone.load_state_dict(torch.load("weight/backbone.pth", map_location="cpu"))
encoder.load_state_dict(torch.load("weight/encoder.pth", map_location="cpu"))

backbone.eval()
encoder.eval()

transform = T.Compose([
    T.Resize((640, 640)),  
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

@app.post("/encode")
async def encode(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        x = transform(image).unsqueeze(0) # [1, 3, 640, 640]

        with torch.no_grad():
            backbone_out = backbone(x) 
            feat = encoder(backbone_out)
            
        buffer = io.BytesIO()
        torch.save(feat, buffer)
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="application/octet-stream")
    except Exception as e:
        return Response(content=str(e).encode(), media_type="text/plain", status_code=500)