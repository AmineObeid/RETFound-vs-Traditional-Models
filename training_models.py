import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
INPUT_SIZE = 256  
BATCH_SIZE = 16   
EPOCHS = 10   
LR = 1e-3  

data_paths = {
    "train": "./data/train",  
    "val": "./data/val",   
    "test": "./data/test"  
}

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)), #unncessary since we preserve the size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet values
])

train_dataset = datasets.ImageFolder(root=data_paths["train"], transform=transform)
val_dataset = datasets.ImageFolder(root=data_paths["val"], transform=transform)
test_dataset = datasets.ImageFolder(root=data_paths["test"], transform=transform)

train_loader = DataLoader(train_dataset, BATCH_SIZE=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, BATCH_SIZE=BATCH_SIZE, shuffle=False)

def load_model(configuration):
    name = configuration["name"]
    if configuration["source"] == "timm":
        model = timm.create_model(name, pretrained=configuration.get("pretrained", True), NUM_CLASSES=NUM_CLASSES)
    elif configuration["source"] == "torchvision":
        if name == "resnet50":
            model = models.resnet50(pretrained=configuration.get("pretrained", True))
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif name == "densenet121":
            model = models.densenet121(pretrained=configuration.get("pretrained", True))
            model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model = model.to(device)
    return model

model_configs = [
    {"name": "resnet50", "source": "torchvision", "pretrained": True},
    {"name": "densenet121", "source": "torchvision", "pretrained": True},
    {"name": "efficientnet_b0", "source": "timm", "pretrained": True},
]

def train_model(model, train_loader, val_loader, EPOCHS=10, lr=1e-3):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()  
        current_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss += loss

        val_accuracy, val_f1, val_roc_auc = evaluate(model, val_loader)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")  

def evaluate(model, loader):
    model.eval()  
    preds_arr = []
    labels_arr = []
    probabilities_arr = []
    
    with torch.inference_mode():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            max_tensor, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            preds_arr.extend(preds.cpu().numpy())
            labels_arr.extend(target.cpu().numpy())
            probabilities_arr.extend(probabilities.cpu().numpy())

    accuracy = accuracy_score(labels_arr, preds_arr)
    
    f1 = f1_score(labels_arr, preds_arr, average='weighted')
    
    lb = LabelBinarizer()
    labels_arr_bin = lb.fit_transform(labels_arr)
    roc_auc = roc_auc_score(labels_arr_bin, probabilities_arr, multi_class='ovr', average='macro')

    return accuracy, f1, roc_auc

results = {}

for configuration in model_configs:
    model = load_model(configuration)
    
    train_model(model, train_loader, val_loader, EPOCHS=EPOCHS, lr=LR)
    
    model.load_state_dict(torch.load("best_model.pth"))
    
    accuracy, f1, roc_auc = evaluate(model, test_loader)
    results[configuration["name"]] = {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}

results['RETFound'] = {'accuracy': 0.89, 'f1': 0.89, 'roc_auc': 0.98}  

model_names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in model_names]
f1_scores = [results[name]["f1"] for name in model_names]
roc_aucs = [results[name]["roc_auc"] for name in model_names]

colors = ['skyblue', 'salmon', 'lightgreen', 'gold']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].bar(model_names, accuracies, color=colors)
axes[0].set_title("Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_xticklabels(model_names, rotation=45)

axes[1].bar(model_names, f1_scores, color=colors)
axes[1].set_title("F1 Score")
axes[1].set_ylabel("F1 Score")
axes[1].set_xticklabels(model_names, rotation=45)

axes[2].bar(model_names, roc_aucs, color=colors)
axes[2].set_title("ROC-AUC")
axes[2].set_ylabel("ROC-AUC")
axes[2].set_xticklabels(model_names, rotation=45)

plt.tight_layout()
plt.savefig("model_comparison_metrics.png", dpi=300)  
plt.show()

