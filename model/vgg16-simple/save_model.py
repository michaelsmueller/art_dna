import requests

file_path = "paint1.jpeg"
response = requests.post("http://localhost:8000/predict", files={"image": open(file_path, "rb")})

print("Status Code:", response.status_code)
print("Response:", response.text)
