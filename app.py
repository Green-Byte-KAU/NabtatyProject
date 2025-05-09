from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

class PalmDiseaseClassifier(nn.Module):
    def __init__(self, number_of_classes=9):
        super(PalmDiseaseClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128*28*28, number_of_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        # print(x.shape) # to get the needed input size for fc1
        
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.drop1(x)
        x = self.fc1(x)
        
        return x


app = Flask(__name__)

CORS(app)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = PalmDiseaseClassifier(number_of_classes=9)
model.load_state_dict(torch.load("best_palmdisease_cnn.pth"))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_labels = ['Black Scorch', 'Fusarium Wilt', 'Healthy sample', 'Leaf Spots', 'Magnesium Deficiency',
                'Manganese Deficiency', 'Parlatoria Blanchardi', 'Potassium Deficiency', 'Rachis Blight']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image)
        
        with torch.no_grad():
            image = image.to(device)
            image = image.unsqueeze(0)
            output = model(image)
            _, predicted = torch.max(output, 1)
            confidence,_ = torch.max(F.softmax(output,1),1)
            
            predicted_class = class_labels[predicted.item()]
            print(f"Prediction: {predicted_class}")
            print(f"confidence = %{round(confidence.item()*100,1)}")

        return jsonify({
            'prediction': predicted_class,
            'confidence': '%'+str(round(confidence.item()*100,1)),
            'status': 'success'
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)