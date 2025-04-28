from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

class PalmClassifier(nn.Module):
    def __init__(self, number_of_classes=9):
        super(PalmClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(128, number_of_classes)
        
        self.drop1 = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.gap(x)
        
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = self.fc1(x)
        
        return x


app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(num_classes=9)
model.load_state_dict(torch.load("C:\\Users\\ahmad\\Desktop\\ai\\date_palm_leaf_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_labels = ['Black Scorch', 'Fusarium Wilt', 'Healthy sample', 'Leaf Spots', 'Magnesium Deficiency', 'Manganese Deficiency', 'Parlatoria Blanchardi', 'Potassium Deficiency', 'Rachis Blight']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)