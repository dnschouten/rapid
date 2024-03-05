import numpy as np
import torch
import cv2
import kornia as K
from typing import List, Any
from lightglue.utils import rbd
import torch.nn.functional as F

# import demo_utils


def get_keypoints(detector: Any, matcher: Any, detector_name: str, ref_image: np.ndarray, moving_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function to get keypoints from any of the supported detectors.
    """

    if detector_name in ["superpoint", "disk"]:
        ref_points, moving_points, scores = get_lightglue_keypoints(ref_image, moving_image, detector, matcher)
    elif detector_name == "dalf":
        ref_points, moving_points = get_dalf_keypoints(ref_image, moving_image, detector, matcher)
    elif detector_name == "loftr":
        ref_points, moving_points, scores = get_loftr_keypoints(ref_image, moving_image, matcher)
    elif detector_name == "aspanformer":
        ref_points, moving_points, scores = get_aspanformer_keypoints(ref_image, moving_image, matcher)
    elif detector_name == "roma":
        ref_points, moving_points, scores = get_roma_keypoints(ref_image, moving_image, matcher)

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
    ref_points = np.float32([i.astype("int") for i in ref_features['keypoints'][matches[..., 0]].cpu().numpy()])
    moving_points = np.float32([i.astype("int") for i in moving_features['keypoints'][matches[..., 1]].cpu().numpy()])

    return ref_points, moving_points, matches01["scores"].detach().cpu().numpy()


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
    with torch.no_grad():
        matches = matcher(input)

    ref_points = matches["keypoints0"].cpu().numpy()
    moving_points = matches["keypoints1"].cpu().numpy()
    scores = matches["confidence"].cpu().numpy()

    # Rescale to original size
    ref_points = ref_points * (ref_image.shape[0] / LOFTR_SIZE)
    moving_points = moving_points * (moving_image.shape[0] / LOFTR_SIZE)

    return ref_points, moving_points, scores


def get_dalf_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with DALF
    """

    # Extract features
    ref_points, desc_ref = detector.detectAndCompute(ref_image)
    moving_points, desc_moving = detector.detectAndCompute(moving_image)
    
    # Match features
    matches = matcher.match(desc_ref, desc_moving)

    # Get matching keypoints
    ref_points = np.float32([ref_points[m.queryIdx].pt for m in matches])
    moving_points = np.float32([moving_points[m.trainIdx].pt for m in matches])

    return ref_points, moving_points


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
    with torch.no_grad():
        matcher(data, online_resize=True)
        ref_points = data['mkpts0_f'].cpu().numpy()
        moving_points = data['mkpts1_f'].cpu().numpy()

    # Convert keypoints back to original shape
    ref_points = ref_points * factor
    moving_points = moving_points * factor
    scores = data['mconf'].cpu().numpy()

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

    # Match images directly
    warp, certainty = matcher.match(ref_path, moving_path)

    # Convert images to tensors 
    ref_tensor = (torch.tensor(ref_image)/ 255).to("cuda").permute(2, 0, 1)
    moving_tensor = (torch.tensor(moving_image) / 255).to("cuda").permute(2, 0, 1)
    
    # Warp images directly according to features
    H, W = matcher.get_output_resolution()
    im1_transfer_rgb = F.grid_sample(
        ref_tensor[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    im2_transfer_rgb = F.grid_sample(
        moving_tensor[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]

    # Created combined image with masked out uncertain areas
    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)
    white_im = torch.ones((H, 2*W), device="cuda")
    vis_im = certainty * warp_im + (1 - certainty) * white_im

    return 
