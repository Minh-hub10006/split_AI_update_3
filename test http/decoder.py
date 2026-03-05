from fastapi import FastAPI
app=FastAPI()
@app.post("/decode")
def decode(data:dict):
    number = data["number"]
    result=number+10
    return {"decoded": result}
