import torch
from insightface.insight_face import iresnet100
from torchvision import transforms
from face_detection import device

# Load InsightFace model
model_emb = iresnet100()

# Load the model weights
weight = torch.load("insightface/resnet100_backbone.pth", map_location=device)
model_emb.load_state_dict(weight)

# Set the model to use the specified device (CPU or GPU)
model_emb.to(device)

# Set the model to evaluation mode
model_emb.eval()

# Define the preprocessing transformation for face images
face_preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Resize((112, 112)),  # Resize to (112, 112)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Initialize variables
score = 0
name = None