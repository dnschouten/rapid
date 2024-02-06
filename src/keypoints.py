import numpy as np
import torch
import cv2
import kornia as K
from typing import List, Any
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd
from modules.models.DALF import DALF_extractor as DALF


def get_keypoints(detector: str, ref_image: np.ndarray, moving_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapped function to get either LightGlue or DALF keypoints
    """

    assert detector.lower() in ["lightglue", "dalf"]

    if detector.lower() == "lightglue":
        points_ref, points_moving = get_lightglue_keypoints(ref_image, moving_image)
    elif detector.lower() == "dalf":
        points_ref, points_moving = get_dalf_keypoints(ref_image, moving_image)

    return points_ref, points_moving


def get_lightglue_keypoints(ref_image: np.ndarray, moving_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with LightGlue
    """

    # Initialize lightglue detector and matcher
    lightglue_detector = SuperPoint(max_num_keypoints=None).eval().cuda()  
    lightglue_matcher = LightGlue(features='superpoint').eval().cuda() 

    # Convert images to tensor
    ref_tensor = torch.tensor(ref_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
    moving_tensor = torch.tensor(moving_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()

    # Extract features
    ref_features = lightglue_detector.extract(ref_tensor)
    moving_features = lightglue_detector.extract(moving_tensor)

    # Find matches with features from reference image
    matches01 = lightglue_matcher({'image0': ref_features, 'image1': moving_features})
    ref_features2, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
    matches = matches01['matches']

    # Get matching keypoints
    points_ref = np.float32([i.astype("int") for i in ref_features2['keypoints'][matches[..., 0]].cpu().numpy()])
    points_moving = np.float32([i.astype("int") for i in moving_features['keypoints'][matches[..., 1]].cpu().numpy()])

    return points_ref, points_moving


def get_loftr_keypoints(ref_image: np.ndarray, moving_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    ref_tensor = K.resize(ref_tensor, (480, 640), antialias=True).cuda()
    moving_tensor = K.resize(moving_tensor, (480, 640), antialias=True).cuda()

    # Get matcher and match
    matcher = K.feature.LoFTR(pretrained="indoor_new").eval().cuda()
    input = {"image0": ref_tensor, "image1": moving_tensor}
    with torch.no_grad():
        matches = matcher(input)

    ref_points = matches["keypoints0"].cpu().numpy()
    moving_points = matches["keypoints1"].cpu().numpy()

    return ref_points, moving_points


def get_dalf_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, device: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with DALF
    """

    # Initialize DALF detector and matcher
    dalf_detector = DALF(dev=device)

    # Extract features
    points_ref, desc_ref = dalf_detector.detectAndCompute(ref_image)
    points_moving, desc_moving = dalf_detector.detectAndCompute(moving_image)
    
    # Match features
    matcher = cv2.BFMatcher(crossCheck = True)
    matches = matcher.match(desc_ref, desc_moving)

    # Get matching keypoints
    points_ref = np.float32([points_ref[m.queryIdx].pt for m in matches])
    points_moving = np.float32([points_moving[m.trainIdx].pt for m in matches])

    return points_ref, points_moving
