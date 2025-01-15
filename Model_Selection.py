# Importing Libraries
import torch
import numpy as np
from collections import OrderedDict

# Initializing Parameters
SKIP_LAYERS = ["fc.weight", "fc.bias"]  # Extra layers in base model


# Helper Functions
def EuclideanDist(p1: np.ndarray | list[float], p2: np.ndarray | list[float]) -> float:
    """
    Description
    -----------
    Calculates Eucliedean distance between two points in n-dimensional space.

    Parameters
    ----------
    p1 : np.ndarray | list[float]
        First point in form of a list. i.e. (x,y) or (x,y,z) ...
    p2 : np.ndarray | list[float]
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
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = np.sum(np.power(p1 - p2, 2)) ** 0.5
    return dist


def heronsArea(
    p1: np.ndarray | list[float],
    p2: np.ndarray | list[float],
    p3: np.ndarray | list[float],
) -> float:
    """
    Description
    -----------
    Calculates the area of a triangle in n-dimensional space using Heron's Formula.

    Parameters
    ----------
    p1 : np.ndarray | list[float]
        First point in form of a list. i.e. (x,y) or (x,y,z) ...
    p2 : np.ndarray | list[float]
        Second point in form of a list. i.e. (x,y) or (x,y,z) ...
    p3 : np.ndarray | list[float]
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
    a,b,c = sorted([a,b,c],reverse=True)
    area = 0.25 * (((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))) ** 0.5)
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

