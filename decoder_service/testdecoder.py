from fastapi import FastAPI, Request
from fastapi.responses import Response
import torch
import io
from src.core.yaml_config import YAMLConfig

app = FastAPI()
cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model

decoder = model.decoder

decoder.load_state_dict(
    torch.load("weight/decoder.pth", map_location="cpu")
)

decoder.eval()

@app.post("/decode")
async def decode(request: Request):
    try:
        body = await request.body()
        buffer = io.BytesIO(body)
        feat = torch.load(buffer, map_location="cpu", weights_only=False)

        with torch.no_grad():
            output = model.decoder(feat)

        buffer_out = io.BytesIO()
        torch.save(output, buffer_out)
        buffer_out.seek(0)

        return Response(content=buffer_out.getvalue(), media_type="application/octet-stream")
    except Exception as e:
        return Response(content=str(e).encode(), media_type="text/plain", status_code=500)