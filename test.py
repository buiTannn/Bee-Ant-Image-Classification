import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)  
num_classes = 2  
model.fc = torch.nn.Linear(512, num_classes)  

# Load state_dict tu file
state_dict_path = "./models/best_model.pth"
if os.path.exists(state_dict_path):
    state_dict = torch.load(state_dict_path, map_location=device) 
    new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)  
    print("Mo hinh da duoc tai thanh cong!")
else:
    print("Khong tim thay file best_model.pth!")
    exit()

model.to(device) 
model.eval()  

# Dinh nghia pipeline 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Khong tim thay anh: {img_path}")
        return None

    img = Image.open(img_path).convert("RGB") 
    img = transform(img).unsqueeze(0).to(device) 
    
    with torch.no_grad(): 
        output = model(img)
        _, predicted = torch.max(output, 1)  
    
    return predicted.item()

img_path = "bee3.jpg"  
predicted_class = predict_image(img_path)

if predicted_class is not None:
    print(f"Predicted Class: {predicted_class}")
