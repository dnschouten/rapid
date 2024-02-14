import numpy as np
import torch
import cv2
import kornia as K
import time
from typing import List, Any
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd
from modules.models.DALF import DALF_extractor as DALF


def get_keypoints(detector: Any, matcher: Any, detector_name: str, ref_image: np.ndarray, moving_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapped function to get keypoints from any of the supported detectors.
    """

    if detector_name in ["superpoint", "disk"]:
        points_ref, points_moving, scores = get_lightglue_keypoints(ref_image, moving_image, detector, matcher)
    elif detector_name == "dalf":
        points_ref, points_moving = get_dalf_keypoints(ref_image, moving_image, detector, matcher)
    elif detector_name == "loftr":
        points_ref, points_moving, scores = get_loftr_keypoints(ref_image, moving_image, matcher)

    return points_ref, points_moving, scores


def get_lightglue_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with LightGlue
    """

    # Convert images to tensor
    ref_tensor = torch.tensor(ref_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
    moving_tensor = torch.tensor(moving_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()

    # Extract features and match
    with torch.inference_mode():
        ref_features = detector.extract(ref_tensor)
        moving_features = detector.extract(moving_tensor)
        matches01 = matcher({'image0': ref_features, 'image1': moving_features})

    # Convert
    ref_features, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
    matches = matches01['matches']

    # Get matching keypoints
    points_ref = np.float32([i.astype("int") for i in ref_features['keypoints'][matches[..., 0]].cpu().numpy()])
    points_moving = np.float32([i.astype("int") for i in moving_features['keypoints'][matches[..., 1]].cpu().numpy()])

    return points_ref, points_moving, matches01["scores"].detach().cpu().numpy()


def get_loftr_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    First try with the LoFTR keypoint detector and matcher.
    """

    # Convert to grayscale
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
    moving_image = cv2.cvtColor(moving_image, cv2.COLOR_RGB2GRAY)

    # Convert to tensor with Kornia
    ref_tensor = K.image_to_tensor(ref_image, None).float() / 255.0
    moving_tensor = K.image_to_tensor(moving_image, None).float() / 255.0

    # Rescale to fit LoFTR architecture
    ref_tensor = K.geometry.transform.resize(ref_tensor, (480, 480), antialias=True).cuda()
    moving_tensor = K.geometry.transform.resize(moving_tensor, (480, 480), antialias=True).cuda()

    # Get matcher and match
    input = {"image0": ref_tensor, "image1": moving_tensor}
    with torch.no_grad():
        matches = matcher(input)

    ref_points = matches["keypoints0"].cpu().numpy()
    moving_points = matches["keypoints1"].cpu().numpy()
    scores = matches["confidence"].cpu().numpy()

    return ref_points, moving_points, scores


def get_dalf_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with DALF
    """

    # Extract features
    points_ref, desc_ref = detector.detectAndCompute(ref_image)
    points_moving, desc_moving = detector.detectAndCompute(moving_image)
    
    # Match features
    matches = matcher.match(desc_ref, desc_moving)

    # Get matching keypoints
    points_ref = np.float32([points_ref[m.queryIdx].pt for m in matches])
    points_moving = np.float32([points_moving[m.trainIdx].pt for m in matches])

    return points_ref, points_moving
