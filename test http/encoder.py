from fastapi import FastAPI
app = FastAPI()
@app.post("/encode")
def encode(data:dict):
    number=data["number"]
    result=number*2
    return {"encoded":result}
