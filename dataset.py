import torch

# Load the model
model_path = "model.pt"
model = torch.load(model_path)

# Inspect the class names
if 'names' in model:
    print("Class names:", model['names'])
else:
    print("The model does not contain 'names'.")
