from fastapi.testclient import TestClient
from app.main import app
import numpy as np
import cv2
import json

client = TestClient(app)

# Create a dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite("data/test_images/dummy.jpg", img)

print("Pinging analyze endpoint...")
try:
    with open("data/test_images/dummy.jpg", "rb") as f:
        files = {"file": ("dummy.jpg", f, "image/jpeg")}
        data = {
            "allowed_activities": json.dumps(["walking"]),
            "unauthorized_activities": json.dumps(["running"])
        }
        response = client.post("/api/v1/analyze", files=files, data=data)
        
    print("Status Code:", response.status_code)
    try:
        print("Response payload:", json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print("Response payload:", response.text)
except Exception as e:
    print(f"Failed to connect: {e}")
