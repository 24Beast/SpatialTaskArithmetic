# Importing Libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torchvision.models as torchModels

try:
    from torch_pca import PCA
except ImportError:
    print("torch_pca module unavailable, switching to sklearn version.")
    from sklearn.decomposition import PCA


# Initializing Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATHS = "C:/Users/btokas/Projects/TaskArith/models/"
BASE_MODEL = torchModels.resnet34(weights="IMAGENET1K_V1").to(DEVICE)
NUM_COMPONENTS = 10
SKIP_LAYERS = ["fc.weight", "fc.bias"]  # Extra layers in base model


# Helper Functions
def modelDictToTensor(model_state_dict: OrderedDict) -> torch.Tensor:
    layer_weights = []
    for name, weights in model_state_dict.items():
        if name in SKIP_LAYERS:
            continue
        layer_weights.append(torch.flatten(weights))
    return torch.cat(layer_weights)


# Loading Model Data
BASE_TENSOR = modelDictToTensor(BASE_MODEL.state_dict())
model_types = os.listdir(MODEL_PATHS)
shift_tensors = []
labels = []
for model_type in model_types:
    model_paths = os.listdir(os.path.join(MODEL_PATHS, model_type))
    model_paths = sorted(model_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    for num, model_path in enumerate(model_paths):
        model = torch.load(os.path.join(MODEL_PATHS, model_type, model_path))
        model_tensor = modelDictToTensor(model)
        shift_tensors.append(model_tensor - BASE_TENSOR)
        labels.append(model_type + "_" + model_path)
shift_tensors = torch.stack(shift_tensors)


# Applying PCA
PCA_model = PCA(n_components=NUM_COMPONENTS)
shift_compressed = PCA_model.fit_transform(shift_tensors)[:, :2]
print(
    f"Explained_variance_ratio for first {NUM_COMPONENTS} : {PCA_model.explained_variance_ratio_}"
)

BASE_compressed = PCA_model.transform(torch.zeros((1, *BASE_TENSOR.shape)).to(DEVICE))[
    0, :2
].cpu()

# Visualizing PCA
type_labels = [item.split("_encoder")[0] for item in labels]
type_names, type_counts = np.unique(type_labels, return_counts=True)
shift_compressed = shift_compressed.cpu()
type_splits = torch.split(shift_compressed, type_counts.tolist())
for num in range(len(type_names)):
    x, y = type_splits[num].split(1, dim=1)
    plt.scatter(x, y)
    plt.plot(x, y, label=type_names[num])
plt.scatter(BASE_compressed[0], BASE_compressed[1], label="ImageNet-pretrained")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


# Measuring Deviation
print("BASE Deviations")
type_splits_tensors = shift_tensors.split(type_counts.tolist())
deviations = {}
for num in range(len(type_names)):
    curr_split = type_splits_tensors[num]
    rmses = torch.sqrt(torch.mean(curr_split**2, axis=1))
    deviations[type_names[num]] = rmses
    print(f"{type_names[num]=}")
    print(f"{rmses.min().item()=}, {rmses.max().item()=}, {rmses.mean().item()=}")
    print(f"{rmses=}")

# Measuring Deviation in Compressed Representation
print("\n\nCompressed Deviations:")
compressed_deviations = {}
for num in range(len(type_names)):
    curr_split = type_splits[num]
    rmses = torch.sqrt(torch.mean((curr_split - BASE_compressed) ** 2, axis=1))
    compressed_deviations[type_names[num]] = rmses
    print(f"{type_names[num]=}")
    print(f"{rmses.min().item()=}, {rmses.max().item()=}, {rmses.mean().item()=}")
    print(f"{rmses=}")
