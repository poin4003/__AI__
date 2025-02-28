from transformers import ViTFeatureExtractor, ViTForImageClassification 
from PIL import Image

def load_classification_model():
    model_name = "facebook/deit-base-distilled-patch16-224"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, size=224)
    model = ViTForImageClassification.from_pretrained(model_name)
    return feature_extractor, model

def classify_image(image_path, feature_extractor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]