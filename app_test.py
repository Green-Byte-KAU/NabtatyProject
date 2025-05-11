import unittest
import torch
from flask import json
from app import app, PalmDiseaseClassifier, class_labels, model

class TestPalmDiseaseAPI(unittest.TestCase):
    def setUp(self):
        """Set up test client and mock data."""
        app.testing = True
        self.client = app.test_client()
        
    def test_model_architecture(self):
        """Test that the model can instantiate and process input."""
        
        test_input = torch.randn(1, 3, 224, 224)
        test_input = test_input.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        output = model(test_input)
        self.assertEqual(output.shape[1], len(class_labels))
        
    def test_predict_endpoint_no_file(self):
        """Test /predict with no file uploaded."""
        response = self.client.post('/predict')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'No file uploaded')

    def test_class_labels_consistency(self):
        """Test labels match model output dimensions."""
        model = PalmDiseaseClassifier(number_of_classes=9)
        self.assertEqual(model.fc1.out_features, len(class_labels))

if __name__ == '__main__':
    unittest.main()