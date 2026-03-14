import torch
import pika
import pickle
import cv2
import numpy as np
from src.core.yaml_config import YAMLConfig

cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model

decoder = model.decoder
decoder.load_state_dict(torch.load("weight/decoder.pth", map_location="cpu"))

decoder.eval()


def run_decoder(feats):

    with torch.no_grad():
        outputs = decoder(feats)

    return outputs


def draw_boxes(image, boxes, scores, threshold=0.5):

    img = image.copy()

    h, w = img.shape[:2]

    for box, score in zip(boxes, scores):

        if score < threshold:
            continue

        cx, cy, bw, bh = box

        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(img,
                    f"{score:.2f}",
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1)

    return img


def callback(ch, method, properties, body):

    payload = pickle.loads(body)

    image = payload["image"]
    feats = payload["feature"]

    print("Received image + feature")

    outputs = run_decoder(feats)

    pred_logits = outputs["pred_logits"][0]
    pred_boxes = outputs["pred_boxes"][0]

    scores = pred_logits.softmax(-1).max(-1)[0]

    boxes = pred_boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    result = draw_boxes(image, boxes, scores)

    cv2.imwrite("result.jpg", result)

    print("Saved result.jpg")


connection = pika.BlockingConnection(
    pika.ConnectionParameters("localhost")
)

channel = connection.channel()

channel.queue_declare(queue="feature_queue")

channel.basic_consume(
    queue="feature_queue",
    on_message_callback=callback,
    auto_ack=True
)

print("Waiting for data...")

channel.start_consuming()