import io
from PIL import Image
from flask_cors import CORS
from torch.nn import Linear
from torch import load, device
from torchvision.models import resnet18
from torch.nn.functional import softmax
from flask import Flask, jsonify, request
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# Transform the image to a tensor
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB') 
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

    
# Load the model
model = resnet18(weights='ResNet18_Weights.DEFAULT')
model.fc = Linear(model.fc.in_features, 10)
state_dict = load('model.pth', map_location=device('cpu'))
model.load_state_dict(state_dict['model_state_dict'])
model.eval()

labels = {
    0: "annual_crop",
    1: "forest",
    2: "herbaceous_vegetation",
    3: "highway",
    4: "industrial_building",
    5: "pasture",
    6: "permanent_crop",
    7: "residential_building",
    8: "river",
    9: "sea_lake"
}

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    confidence = softmax(outputs, dim=1)[0][y_hat].item()
    confidences = softmax(outputs, dim=1)[0].tolist()
    return y_hat.item(), confidence, confidences

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img_bytes = file.read()
        class_id, confidence, confidences = get_prediction(image_bytes=img_bytes)
        return jsonify({'class': labels[class_id], 'confidence': confidence, 'confidences': confidences})

@app.route('/')
def index():
    return jsonify({'message': 'Server is running!'})

if __name__ == '__main__':
    app.run()