import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

# Định nghĩa Model
class Model(nn.Module):
    def __init__(self, backbone, n_class):
        super(Model, self).__init__()
        if backbone == "resnet18":
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, n_class)
        elif backbone == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, n_class)
        else:
            raise ValueError("Unsupported backbone. Choose from 'resnet18' or 'vgg16'")

    def forward(self, x):
        return self.model(x)

# Định nghĩa Learner
class Learner:
    def __init__(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, scheduler, workdir, epochs, device):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.workdir = workdir
        self.epochs = epochs
        self.device = device
        self.best_acc = 0.0  # Lưu độ chính xác tốt nhất để lưu model

        os.makedirs(self.workdir, exist_ok=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            epoch_acc = 100 * correct / total
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

            # Lưu model tốt nhất
            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                torch.save(self.model.state_dict(), os.path.join(self.workdir, "best_model_vgg16.pth"))

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        loss_min = 0.0
        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss_min += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc_min = 100 * correct / total
        loss_min /= len(self.test_dataloader)
        print(f"Test Accuracy: {acc_min:.2f}%, Test Loss: {loss_min:.4f}")
        return acc_min, loss_min

    def inference(self, img_path):
        class_names = {0: "ant", 1: "bee"}  
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), # Chuyển ảnh về dạng tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
            _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

# Tạo DataLoader từ Hymenoptera Dataset
def get_dataloaders(batch_size=32):
    data_dir = "./data"
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

# Chạy Training
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_dataloader, test_dataloader = get_dataloaders(batch_size=32)

    # Khởi tạo mô hình
    model = Model("resnet18", 2).to(device)

    # Khởi tạo loss function, optimizer, scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # Khởi tạo Learner
    learner = Learner(model, train_dataloader, test_dataloader, loss_fn, optimizer, scheduler, "./models", 10, device)

    # Train model
    
    learner.train()

    # Test model
    learner.test()

    # Inference thử nghiệm
    output_class = learner.inference("ant.jpg")
    print(f"Predicted Class: {output_class}")
