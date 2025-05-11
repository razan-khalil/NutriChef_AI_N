
# import torch
# from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# class FoodClassifier:
#     def __init__(self,model_name="facebook/food101", device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.extractor = AutoFeatureExtractor.from_pretrained("facebook/food101")
#         self.model = AutoModelForImageClassification.from_pretrained("facebook/food101").to(self.device)

#     def classify_food(self, image):
#         inputs = self.extractor(images=image, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             predicted_label = probs.argmax(dim=-1).item()

#         labels = self.model.config.id2label
#         return labels[predicted_label]
    
    # models/food_classifier.py

import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class FoodClassifier:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").to(self.device)

    def classify_food(self, image):
        inputs = self.extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_label = probs.argmax(dim=-1).item()

        labels = self.model.config.id2label
        return labels[predicted_label]

