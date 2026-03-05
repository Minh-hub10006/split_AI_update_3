import torch
from src.core.yaml_config import YAMLConfig

cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model

checkpoint = torch.load("weight/dfine_l_coco.pth", map_location="cpu")

if "model" in checkpoint:
    state_dict = checkpoint["model"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

# Lưu riêng từng phần
torch.save(model.backbone.state_dict(), "weight/backbone.pth")
torch.save(model.encoder.state_dict(), "weight/encoder.pth")
torch.save(model.decoder.state_dict(), "weight/decoder.pth")

print("Done split weights")