# Importing Libraries
import os
import torch
import numpy as np
from collections import OrderedDict
import torchvision.models as torchModels


# Initializing Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SKIP_LAYERS = ["fc.weight", "fc.bias"]  # Extra layers in base model
MODEL_PATHS = "C:/Users/btokas/Projects/TaskArith/models/"
BASE_MODEL = torchModels.resnet34(weights="IMAGENET1K_V1").to(DEVICE)


# Helper Functions
def EuclideanDist(
    p1: torch.Tensor | list[float], p2: torch.Tensor | list[float]
) -> float:
    """
    Description
    -----------
    Calculates Eucliedean distance between two points in n-dimensional space.

    Parameters
    ----------
    p1 : torch.Tensor | list[float]
        First point in form of a list. i.e. (x,y) or (x,y,z) ...
    p2 : torch.Tensor | list[float]
        Second point in form of a list. i.e. (x,y) or (x,y,z) ...

    Raises
    ------
    ValueError
        Both points must have the same number of dimensions.

    Returns
    -------
    float
        Euclidean distance between the points.
    """
    if len(p1) != len(p2):
        raise ValueError(
            f"Expected p1 and p2 to have same number of dimensions, got {len(p1)} and {len(p2)} dimensional points."
        )
    if type(p1) == list:
        p1 = torch.tensor(p1).to(DEVICE)
    if type(p2) == list:
        p2 = torch.tensor(p2).to(DEVICE)
    dist = torch.sum(torch.pow(p1 - p2, 2)) ** 0.5
    return dist.item()


def heronsArea(
    p1: torch.Tensor | list[float],
    p2: torch.Tensor | list[float],
    p3: torch.Tensor | list[float],
) -> float:
    """
    Description
    -----------
    Calculates the area of a triangle in n-dimensional space using Heron's Formula.

    Parameters
    ----------
    p1 : torch.Tensor | list[float]
        First point in form of a list. i.e. (x,y) or (x,y,z) ...
    p2 : torch.Tensor | list[float]
        Second point in form of a list. i.e. (x,y) or (x,y,z) ...
    p3 : torch.Tensor | list[float]
        Third point in form of a list. i.e. (x,y) or (x,y,z) ...

    Raises
    ------
    ValueError
        All points must have the same number of dimensions.

    Returns
    -------
    float
        Area of the triangle.

    """
    if (len(p1) != len(p2)) | (len(p1) != len(p3)):
        raise ValueError(
            (
                f"Expected p1, p2 and p3 to have same number of dimensions, got {len(p1)}, {len(p2)}, {len(p3)} dimensional points."
            )
        )
    a = EuclideanDist(p1, p2)
    b = EuclideanDist(p2, p3)
    c = EuclideanDist(p1, p3)
    # Numerical stability requires reordering
    a, b, c = sorted([a, b, c], reverse=True)
    area = 0.25 * (
        ((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))) ** 0.5
    )
    if np.isnan(area):
        print("Small values might lead to false warnings.")
        return 0.0
    return area


def modelDictToTensor(model_state_dict: OrderedDict) -> torch.Tensor:
    """
    Description
    -----------
    Converts a pytorch model dict into tensor suitable for task arithmetic operations.

    Parameters
    ----------
    model_state_dict : OrderedDict
        Standard pytorch model dict.

    Returns
    -------
    torch.Tensor
        Tensor containing model weights.

    """
    layer_weights = []
    for name, weights in model_state_dict.items():
        if name in SKIP_LAYERS:
            continue
        layer_weights.append(torch.flatten(weights))
    return torch.cat(layer_weights)


# Loading Models and extracting tensors
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
type_labels = [item.split("_encoder")[0] for item in labels]
type_names, type_counts = np.unique(type_labels, return_counts=True)
type_splits_tensors = shift_tensors.split(type_counts.tolist())


# Getting distance
areas = []
for i in range(type_counts[0]):
    p1 = type_splits_tensors[0][i]
    for j in range(type_counts[1]):
        p2 = type_splits_tensors[1][j]
        for k in range(type_counts[2]):
            p3 = type_splits_tensors[2][k]
            area = heronsArea(p1, p2, p3)
            areas.append(area)
            print(f"{area=}")
