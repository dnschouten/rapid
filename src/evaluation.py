import numpy as np
import SimpleITK as sitk
from radiomics import shape
from typing import Tuple, List
from scipy.ndimage import zoom
import torch
import cv2

from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd
from modules.models.DALF import DALF_extractor as DALF


def compute_sphericity(mask: np.ndarray) -> float:
    """
    Function to compute the sphericity of a mask.
    """

    # Convert to 3D mask if required
    if len(mask.shape) == 4:
        mask = np.mean(mask, axis=-1)

    # Downsample to roughly ~200 width/height
    ds_factor = np.max(mask.shape) // 200
    mask = zoom(mask, 1/ds_factor, order=0)

    # Convert to simpleITK image
    mask = (mask / np.max(mask)).astype("uint8")
    sitk_mask = sitk.GetImageFromArray(mask)

    # Compute sphericity
    shape_features = shape.RadiomicsShape(sitk_mask, sitk_mask, label=1)
    shape_features.enableFeatureByName('Sphericity', True)
    shape_features.execute()
    sphericity = float(shape_features.featureValues['Sphericity'])

    return sphericity


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Function to compute the dice score between two 2D masks.
    """

    assert len(mask1.shape) == 2, "mask1 must be 2 dimensional"
    assert len(mask2.shape) == 2, "mask2 must be 2 dimensional"
    assert len(np.unique(mask1)) == 2, "mask1 must be binary"
    assert len(np.unique(mask2)) == 2, "mask2 must be binary"

    # Normalize
    mask1 = (mask1 / np.max(mask1)).astype("uint8")
    mask2 = (mask2 / np.max(mask2)).astype("uint8")

    # Compute dice
    eps = 1e-6
    intersection = np.sum(mask1 * mask2)
    dice = (2. * intersection) / (np.sum(mask1) + np.sum(mask2) + eps)

    return dice


def compute_reconstruction_dice(masks: List) -> float:
    """
    Function to compute the dice score between all masks in a list.
    """

    # Compute dice between all masks
    dice_scores = []

    for i in range(len(masks)-1):
        dice_scores.append(compute_dice(masks[i], masks[i+1]))

    return np.mean(dice_scores)


def compute_tre_keypoints():
    """
    Function to compute the target registration error between two sets of keypoints
    """



    return


