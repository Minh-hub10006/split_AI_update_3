import requests
r=requests.post(
    "http://localhost:8000/predict",
    json={"number":5}
)
print(f"Status code: {r.status_code}")
print(f"Response Text: {r.text} ")
if r.status_code ==200:
    try:
        data=r.json()
        print(data)
    except Exception as e:
        print("Lỗi giải mã JSON: ",e)
else:
    print("Server lỗi")