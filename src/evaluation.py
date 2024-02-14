import numpy as np
import SimpleITK as sitk
import pathlib
import matplotlib.pyplot as plt
from radiomics import shape
from typing import List, Any
from scipy.ndimage import zoom

from keypoints import get_keypoints


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

    return np.round(sphericity, 3)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray, normalized: bool = False) -> float:
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

    # Normalize by max achievable dice
    if normalized:
        smallest_mask = np.min([np.sum(mask1), np.sum(mask2)])
        biggest_mask = np.max([np.sum(mask1), np.sum(mask2)])
        max_possible_dice = (2. * smallest_mask) / (smallest_mask + biggest_mask + eps)
        dice = dice / max_possible_dice

    return dice


def compute_reconstruction_dice(masks: List) -> float:
    """
    Function to compute the dice score between all masks in a list.
    """

    # Compute dice between all masks
    dice_scores = []

    for i in range(len(masks)-1):
        dice_scores.append(compute_dice(masks[i], masks[i+1]), normalized=False)

    return np.round(np.mean(dice_scores), 3)


def compute_tre_keypoints(images: List, detector: Any, matcher: Any, detector_name: str, level: int, savedir: pathlib.Path, spacing: float) -> float:
    """
    Function to compute the target registration error between two sets of keypoints
    """

    from visualization import plot_tre_per_pair

    tre_per_pair = []

    # Detect keypoints, match and compute TRE
    for c in range(len(images)-1):

        # Get keypoints
        ref_points, moving_points, scores = get_keypoints(
            detector = detector, 
            matcher = matcher,
            detector_name = detector_name,
            ref_image = images[c], 
            moving_image = images[c+1]
        )

        # Keep the top half of most confident matches
        ref_points = ref_points[scores > np.median(scores)]
        moving_points = moving_points[scores > np.median(scores)]

        # Compute average TRE
        tre = np.median(np.linalg.norm(ref_points - moving_points, axis=-1))

        # Scale w.r.t. pixel spacing
        level_spacing = spacing * 2**level
        scaled_tre = tre * level_spacing

        tre_per_pair.append(scaled_tre)

        savepath = savedir.joinpath("evaluation", f"tre_{c}_{c+1}.png")
        plot_tre_per_pair(
            ref_image = images[c], 
            moving_image = images[c+1], 
            ref_points = ref_points, 
            moving_points = moving_points, 
            tre = scaled_tre,
            savepath = savepath
        )

    return int(np.mean(tre_per_pair))


