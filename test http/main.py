from fastapi import FastAPI
import requests

app= FastAPI()

@app.post("/predict")
def predict(data:dict):
    r1=requests.post(
        "http://localhost:8001/encode",
        json={"number": data["number"]}
    )
    encoded_number=r1.json()["encoded"]

    r2=requests.post(
        "http://localhost:8002/decode",
        json={"number": encoded_number}
    )
    decoded_number =r2.json()["decoded"]
    return {"final_result": decoded_number}
