import numpy as np
import torch
import cv2
import kornia as K
import warnings
from typing import List, Any
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=UserWarning, module='torch.*')

from lightglue.utils import rbd


def get_keypoints(detector: Any, matcher: Any, detector_name: str, ref_image: np.ndarray, moving_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function to get keypoints from any of the supported detectors.
    """

    if detector_name == "superpoint":
        ref_points, moving_points, scores = get_lightglue_keypoints(ref_image, moving_image, detector, matcher)
    elif detector_name == "sift":
        ref_points, moving_points, scores = get_sift_keypoints(ref_image, moving_image, detector, matcher)
    elif detector_name == "dalf":
        ref_points, moving_points, scores = get_dalf_keypoints(ref_image, moving_image, detector, matcher)
    elif detector_name == "loftr":
        ref_points, moving_points, scores = get_loftr_keypoints(ref_image, moving_image, matcher)
    elif detector_name == "aspanformer":
        ref_points, moving_points, scores = get_aspanformer_keypoints(ref_image, moving_image, matcher)
    elif detector_name == "roma":
        ref_points, moving_points, scores = get_roma_keypoints(ref_image, moving_image, matcher)
    elif detector_name == "dedode":
        ref_points, moving_points, scores = get_dedode_keypoints(ref_image, moving_image, detector, matcher)

    return ref_points, moving_points, scores


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

    # Extract matches
    ref_features, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
    matches = matches01['matches']

    # Get matching keypoints
    ref_points = np.float32([i.astype("int") for i in ref_features['keypoints'][matches[..., 0]].detach().cpu().numpy()])
    moving_points = np.float32([i.astype("int") for i in moving_features['keypoints'][matches[..., 1]].detach().cpu().numpy()])

    return ref_points, moving_points, matches01["scores"].detach().cpu().numpy()


def get_sift_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with classical SIFT.
    """

    # Convert to grayscale
    ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
    moving_image_gray = cv2.cvtColor(moving_image, cv2.COLOR_RGB2GRAY)

    # Detect keypoints
    ref_points, ref_features = detector.detectAndCompute(ref_image_gray, None)
    moving_points, moving_features = detector.detectAndCompute(moving_image_gray, None)

    # Match keypoints
    matches = matcher.knnMatch(ref_features, moving_features, k=2)

    # Apply Lowes ratio test
    matches_filtered = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            matches_filtered.append(m)

    ref_points = np.float32([ref_points[m.queryIdx].pt for m in matches_filtered])
    moving_points = np.float32([moving_points[m.trainIdx].pt for m in matches_filtered])

    max_distance = np.max([m.distance for m in matches_filtered])
    scores = np.array([1 - m.distance/max_distance for m in matches_filtered])

    return ref_points, moving_points, scores


def get_loftr_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with the LoFTR detector and matcher.
    """

    LOFTR_SIZE = 480

    # Convert to grayscale
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
    moving_image = cv2.cvtColor(moving_image, cv2.COLOR_RGB2GRAY)

    # Convert to tensor with Kornia
    ref_tensor = K.image_to_tensor(ref_image, None).float() / 255.0
    moving_tensor = K.image_to_tensor(moving_image, None).float() / 255.0

    # Rescale to fit LoFTR architecture
    ref_tensor = K.geometry.transform.resize(ref_tensor, (LOFTR_SIZE, LOFTR_SIZE), antialias=True).cuda()
    moving_tensor = K.geometry.transform.resize(moving_tensor, (LOFTR_SIZE, LOFTR_SIZE), antialias=True).cuda()

    # Get matcher and match
    input = {"image0": ref_tensor, "image1": moving_tensor}
    with torch.inference_mode():
        matches = matcher(input)

    ref_points = matches["keypoints0"].detach().cpu().numpy()
    moving_points = matches["keypoints1"].detach().cpu().numpy()
    scores = matches["confidence"].detach().cpu().numpy()

    # Rescale to original size
    ref_points = ref_points * (ref_image.shape[0] / LOFTR_SIZE)
    moving_points = moving_points * (moving_image.shape[0] / LOFTR_SIZE)

    return ref_points, moving_points, scores


def get_dalf_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with DALF
    """

    with torch.inference_mode():
        # Extract features
        ref_points, desc_ref = detector.detectAndCompute(ref_image)
        moving_points, desc_moving = detector.detectAndCompute(moving_image)
        
        # Match features
        matches = matcher.match(desc_ref, desc_moving)

    # Get matching keypoints
    ref_points = np.float32([ref_points[m.queryIdx].pt for m in matches])
    moving_points = np.float32([moving_points[m.trainIdx].pt for m in matches])
    scores = np.ones(ref_points.shape[0])

    return ref_points, moving_points, scores


def get_aspanformer_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with ASpanformer
    """

    # Convert to grayscale
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
    moving_image = cv2.cvtColor(moving_image, cv2.COLOR_RGB2GRAY)

    # Resize to fit ASpanformer architecture
    LONG_DIM = 1024
    factor = LONG_DIM/max(ref_image.shape[:2])
    ref_image = cv2.resize(
        ref_image, 
        (int(ref_image.shape[1]*factor), int(ref_image.shape[0]*factor))
    )
    moving_image = cv2.resize(
        moving_image, 
        (int(moving_image.shape[1]*factor), int(moving_image.shape[0]*factor))
    )

    # Convert to tensor 
    data = {
        "image0":torch.from_numpy(ref_image/255)[None, None].cuda().float(),
        "image1":torch.from_numpy(moving_image/255)[None, None].cuda().float()
    }

    # Extract features and match
    with torch.inference_mode():
        matcher(data, online_resize=True)
        ref_points = data['mkpts0_f'].detach().cpu().numpy()
        moving_points = data['mkpts1_f'].detach().cpu().numpy()

    # Convert keypoints back to original shape
    ref_points = ref_points * factor
    moving_points = moving_points * factor
    scores = data['mconf'].detach().cpu().numpy()

    return ref_points, moving_points, scores


def get_roma_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with ROMA
    """

    # Resize to ROMA scale
    ROMA_SIZE = 560
    ref_image_save = cv2.resize(ref_image, (ROMA_SIZE, ROMA_SIZE))
    moving_image_save = cv2.resize(moving_image, (ROMA_SIZE, ROMA_SIZE))

    # Roma requires image paths instead of images
    ref_path = "/tmp/ref.png"
    moving_path = "/tmp/moving.png"
    cv2.imwrite(ref_path, ref_image_save)
    cv2.imwrite(moving_path, moving_image_save)

    with torch.inference_mode():
        # Match images directly
        warp, certainty = matcher.match(ref_path, moving_path)

        # Convert to pixel coordinates
        matches, certainty = matcher.sample(warp, certainty)
        ref_points , moving_points = matcher.to_pixel_coordinates(matches, ROMA_SIZE, ROMA_SIZE, ROMA_SIZE, ROMA_SIZE)

    # Scale to original pixel size
    upscale = ref_image.shape[0] / ROMA_SIZE
    ref_points = (ref_points * upscale).cpu().numpy()
    moving_points = (moving_points * upscale).cpu().numpy()

    # Only keep matches with high certainty
    ref_points = ref_points[certainty.cpu().numpy() > 0.9]
    moving_points = moving_points[certainty.cpu().numpy() > 0.9]
    scores = certainty.cpu().numpy()[certainty.cpu().numpy() > 0.9]

    return ref_points, moving_points, scores


def get_dedode_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with DeDoDe
    """

    detector, descriptor = detector

    W_A, H_A = ref_image.shape[:2]
    W_B, H_B = moving_image.shape[:2]

    # DeDoDe requires image paths instead of images
    ref_path = "/tmp/ref.png"
    moving_path = "/tmp/moving.png"
    cv2.imwrite(ref_path, ref_image)
    cv2.imwrite(moving_path, moving_image)

    with torch.inference_mode():
        # Fetch keypoints
        detections_A = detector.detect_from_path(ref_path, num_keypoints = 10_000)
        keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]

        detections_B = detector.detect_from_path(moving_path, num_keypoints = 10_000)
        keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]

        # Use decoupled descriptor to describe keypoints
        description_A = descriptor.describe_keypoints_from_path(ref_path, keypoints_A)["descriptions"]
        description_B = descriptor.describe_keypoints_from_path(moving_path, keypoints_B)["descriptions"]

        # Match and convert to pixel coordinates
        matches_A, matches_B, _ = matcher.match(
            keypoints_A, 
            description_A,
            keypoints_B, 
            description_B,
            P_A = P_A,
            P_B = P_B,
            normalize = True, 
            inv_temp=20, 
            threshold = 0.001
        )
        matches_A, matches_B = matcher.to_pixel_coords(matches_A, matches_B, H_A, W_A, H_B, W_B)

    ref_points = matches_A.detach().cpu().numpy()
    moving_points = matches_B.detach().cpu().numpy()
    scores = np.ones(ref_points.shape[0])

    return ref_points, moving_points, scores

