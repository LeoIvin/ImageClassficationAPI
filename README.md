# Image Classification API using FastAPI and ResNet50
This is a FastAPI project for building an image classification API using the ResNet50 model. The API allows users to upload an image and receive predictions about its content.

# Setup and Installation
1. Install Dependencies:
Ensure you have Python installed on your machine. Install the required Python packages using the following:
`pip install fastapi uvicorn[standard] pillow numpy tensorflow`

3. Run the API:
Save the provided code in a file (e.g., app.py). Open a terminal and run the following command:
`uvicorn app:app --reload`
The API will be accessible at `http://127.0.0.1:8000`.

# Usage 
1. Home Endpoint:
Navigate to `http://127.0.0.1:8000/` in your web browser or use a tool like curl to test:
`curl http://127.0.0.1:8000/`

2. Prediction Endpoint:
Use the /predict endpoint to upload an image and receive predictions. You can use curl as follows:
`curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"`

# Example Usage (requests library)

```
import requests

# Specify the API endpoint
api_url = "http://127.0.0.1:8000/predict"

# Open the image file
files = {'file': ('lion.jpg', open('path/to/your/lion.jpg', 'rb'))}

# Make the prediction request
response = requests.post(api_url, files=files)

# Print the response
print(response.json())
```


 
