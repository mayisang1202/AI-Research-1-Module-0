import torch
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        break 

accuracy = (correct / total) * 100
print(f"ResNet on MNIST Accuracy: {accuracy:.2f}%")