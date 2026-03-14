import torch
import pika
import io
import pickle,sys
import torchvision.transforms as T
from PIL import Image
import numpy as np
from src.core.yaml_config import YAMLConfig

cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model

backbone = model.backbone
encoder = model.encoder

backbone.load_state_dict(torch.load("weight/backbone.pth", map_location="cpu"))
encoder.load_state_dict(torch.load("weight/encoder.pth", map_location="cpu"))

backbone.eval()
encoder.eval()

transform = T.Compose([
    T.Resize((640,640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

def load_image(path):

    img = Image.open(path).convert("RGB")

    img_np = np.array(img)

    tensor = transform(img).unsqueeze(0)

    return img_np, tensor


def run_encoder(x):

    with torch.no_grad():

        feats = backbone(x)
        memory = encoder(feats)

    return memory


def send_payload(payload):

    connection = pika.BlockingConnection(
        pika.ConnectionParameters("localhost")
    )

    channel = connection.channel()
    channel.queue_declare(queue="feature_queue")

    data = pickle.dumps(payload)

    channel.basic_publish(
        exchange="",
        routing_key="feature_queue",
        body=data
    )

    connection.close()

    print("Payload sent")


img_np, x = load_image("test.jpg")

memory = run_encoder(x)

payload = {
    "image": img_np,
    "feature": memory
}

print("Feature tensors:", len(memory))

send_payload(payload)
# size image
image_size = sys.getsizeof(img_np)

# size feature (serialize thử)
feature_bytes = pickle.dumps(memory)
feature_size = len(feature_bytes)

# size payload
payload_bytes = pickle.dumps(payload)
payload_size = len(payload_bytes)

print("Image size:", image_size/1024, "KB")
print("Feature size:", feature_size/1024, "KB")
print("Total payload:", payload_size/1024, "KB")