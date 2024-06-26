import cv2
import numpy as np
import pyvips
import torch
import sys

from skimage import transform
from skimage.transform import EuclideanTransform
from skimage.measure import ransac
from typing import List, Any

import pymagsac
sys.path.append("/detectors/DALF_CVPR_2023")
from modules.tps import RANSAC
from modules.tps import pytorch as tps_pth
from modules.tps import numpy as tps_np


def apply_affine_magsac(moving_points: np.ndarray, ref_points: np.ndarray) -> tuple([np.ndarray, np.ndarray, Any]):
    """
    Function to apply MAGSAC to filter plausible matches for affine transform.
    """

    # Convert to 3D points
    ref_points_3d = np.hstack((ref_points, np.ones((len(ref_points), 1))))    
    moving_points_3d = np.hstack((moving_points, np.ones((len(moving_points), 1))))
    matches = np.ascontiguousarray(np.hstack((ref_points_3d, moving_points_3d)))
    probs = np.ones((len(matches), 1))

    # Apply MAGSAC to further filter plausible matches
    mat, inliers = pymagsac.findRigidTransformation(
        matches, 
        probabilities = probs,
        use_magsac_plus_plus = True,
        sampler=0,
        # sigma_th = 0.95
    )

    if np.sum(inliers) > 0:
        # Convert to proper 3x3 matrix
        matrix = np.ones((3, 3))
        matrix[:2, :2] = -mat.T[:2, :2]
        matrix[:2, 2] = -mat.T[:2, 3]
        mat = EuclideanTransform(matrix = matrix)
    else:
        mat = EuclideanTransform(rotation = 0, translation = 0)
        inliers = [True] * len(ref_points)
        # print(f"unable to fit magsac++, returning identity matrix")

    # Filter based on inliers
    ref_points = ref_points[inliers]
    moving_points = moving_points[inliers]

    return ref_points, moving_points, inliers, mat


def apply_affine_ransac(moving_points: np.ndarray, ref_points: np.ndarray, image: np.ndarray, ransac_thres: float) -> tuple([np.ndarray, np.ndarray, Any]):
    """
    Function to apply RANSAC to filter plausible matches for affine transform.
    """

    min_matches = 10
    inliers = np.array([False] * len(moving_points))
    res_thres = int(image.shape[0] * ransac_thres)

    # Default to identity matrix 
    model = EuclideanTransform(rotation = 0, translation = 0)

    # Apply ransac to further filter plausible matches
    if len(moving_points) > min_matches:

        model, inliers = ransac(
            (moving_points, ref_points),
            EuclideanTransform, 
            min_samples=min_matches,
            residual_threshold=res_thres,
            max_trials=1000
        )

        # Case where convergence fails
        if not isinstance(inliers, np.ndarray):
            inliers = np.array([False] * len(moving_points))

    # Filter matches based on RANSAC if there are enough inliers
    if isinstance(inliers, np.ndarray) and np.sum(inliers) > min_matches:
        ref_points = np.float32([p for p, i in zip(ref_points, inliers) if i])
        moving_points = np.float32([p for p, i in zip(moving_points, inliers) if i])

    return ref_points, moving_points, inliers, model


def estimate_affine_transform(moving_points: np.ndarray, ref_points: np.ndarray, image: np.ndarray, ransac: bool, ransac_thres: float) -> tuple([np.ndarray, List]):
    """
    Function to estimate an affine transform between two sets of points.
    """

    if len(moving_points) > 0:
        # Filter matches based on RANSAC
        if ransac:
            ref_points_m, moving_points_m, inliers, mat1 = apply_affine_magsac(
                moving_points = moving_points, 
                ref_points = ref_points
            )
            ref_points_r, moving_points_r, inliers, mat2 = apply_affine_ransac(
                moving_points = moving_points, 
                ref_points = ref_points, 
                image = image, 
                ransac_thres = ransac_thres
            )

            # Compare magsac and ransac
            # print(f"magsac: rot={mat1.rotation:2f}, trans=[{mat1.translation[0]:2f}, {mat1.translation[1]:2f}]")
            # print(f"ransac: rot={mat2.rotation:2f}, trans=[{mat2.translation[0]:2f}, {mat2.translation[1]:2f}]")

        else:
            inliers = np.array([True] * len(moving_points))

            # Estimate limited affine transform with only rotation and translation
        matrix = transform.estimate_transform(
            "euclidean", 
            moving_points_m, 
            ref_points_m
        )
        
    # Return identity matrix if no keypoints are found
    else:
        matrix = transform.EuclideanTransform(rotation = 0, translation = 0)
        inliers = []
    
    return matrix, np.sum(inliers)


def apply_affine_transform(image: np.ndarray, mask: np.ndarray, tform: np.ndarray) -> tuple([np.ndarray, Any]):
    """
    Apply an affine transform to an image and mask.
    """

    assert len(image.shape) == 3, "image must be 3 dimensional"

    # Warp the main image
    rows, cols, _ = image.shape
    image_warped = cv2.warpAffine(image, tform, (cols, rows), borderValue=(255, 255, 255))

    # Warp mask if available 
    if type(mask) == np.ndarray:
        mask_warped = cv2.warpAffine(mask, tform, (cols, rows), borderValue=(0, 0, 0))
        mask_warped = ((mask_warped > 128)*255).astype("uint8")
    else:
        mask_warped = None

    return image_warped, mask_warped


def apply_affine_transform_fullres(image: pyvips.Image, mask: pyvips.Image, rotation: float, translation: float, center: tuple, scaling: float) -> tuple([pyvips.Image, Any]):
    """
    Apply an affine transform to an image and mask.
    """

    # Get upscaled transformation matrix
    center = (float(center[0] * scaling), float(center[1] * scaling))
    translation = (translation[0] * scaling, translation[1] * scaling)
    tform = cv2.getRotationMatrix2D(center, rotation, 1)
    tform[:, 2] += translation

    # Warp the main image
    image_warped = image.affine(
        (tform[0, 0], tform[0, 1], tform[1, 0], tform[1, 1]),
        interpolate = pyvips.Interpolate.new("bicubic"),
        odx = tform[0, 2],
        ody = tform[1, 2],
        oarea = (0, 0, image.width, image.height),
        background = 255
    )

    # Warp mask if available 
    if type(mask) == pyvips.Image:
        mask_warped = mask.affine(
            (tform[0, 0], tform[0, 1], tform[1, 0], tform[1, 1]),
            interpolate = pyvips.Interpolate.new("nearest"),
            odx = tform[0, 2],
            ody = tform[1, 2],
            oarea = (0, 0, mask.width, mask.height),
            background = 0
        )
    else:
        mask_warped = None

    return image_warped, mask_warped


def apply_deformable_ransac(moving_points: np.ndarray, ref_points: np.ndarray, device: Any, ransac_thres_deformable: float = 0.05) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    """
    Function to apply RANSAC to filter plausible matches for deformable transform.
    """

    # Apply ransac to further filter plausible matches
    inliers = RANSAC.nr_RANSAC(ref_points, moving_points, device, thr = ransac_thres_deformable)

    ref_points = np.float32([p for p, i in zip(ref_points, inliers) if i])
    moving_points = np.float32([p for p, i in zip(moving_points, inliers) if i])

    return ref_points, moving_points


def estimate_deformable_transform(moving_image: np.ndarray, ref_image: np.ndarray, moving_points: np.ndarray, ref_points: np.ndarray, ransac: bool, ransac_thres_deformable: float, deformable_level: int, keypoint_level: int, device: Any) -> tuple([pyvips.Image, Any]):
    """
    Function to estimate the parameters for the deformable transform.
    """

    if ransac:
        ref_points, moving_points = apply_deformable_ransac(moving_points, ref_points, ransac_thres_deformable, device)

    # Get image shapes
    h1, w1 = ref_image.shape[:2]
    h2, w2 = moving_image.shape[:2]
    
    # Normalize coordinates
    c_ref = np.float32(ref_points) / np.float32([w1,h1])
    c_moving = np.float32(moving_points) / np.float32([w2,h2])

    # Downsample image to prevent OOM in deformable grid
    downsample = 2 ** (deformable_level - keypoint_level)
    moving_image_ds = cv2.resize(moving_image, (w2//downsample, h2//downsample), interpolation=cv2.INTER_AREA)

    # Compute theta from coordinates
    moving_image_ds = torch.tensor(moving_image_ds).to(device).permute(2,0,1)[None, ...].float()
    theta = tps_np.tps_theta_from_points(c_ref, c_moving, reduced=True, lambd=0.1)
    theta = torch.tensor(theta).to(device)[None, ...]

    # Create downsampled grid to sample from
    grid = tps_pth.tps_grid(theta, torch.tensor(c_moving, device=device), moving_image_ds.shape)

    # Upsample grid to accomodate original image
    dx = grid.cpu().numpy()[0, :, :, 0]
    dx = ((dx + 1) / 2) * (w2 - 1)

    dy = grid.cpu().numpy()[0, :, :, 1] 
    dy = ((dy + 1) / 2) * (h2 - 1)

    # Upsample using affine rather than resize to account for shape rounding errors
    dx = pyvips.Image.new_from_array(dx).resize(downsample)
    dy = pyvips.Image.new_from_array(dy).resize(downsample)

    # Ensure deformation field is exactly as large as the image. Discepancies can
    # occur due to rounding errors in the shape of the image.
    if (dx.width != w2) or (dx.height != h2):
        dx = dx.gravity("centre", w2, h2)
        dy = dy.gravity("centre", w2, h2)

    index_map = dx.bandjoin([dy])

    return index_map, grid


def apply_deformable_transform(moving_image: np.ndarray, moving_mask: np.ndarray, index_map: pyvips.Image) -> tuple([np.ndarray, np.ndarray]):
    """
    Function to apply the deformable transform. We still use
    pyvips for the actual transform as torch can at most handle ~1000x1000 
    transforms and we need these larger images for downstream tasks.
    """

    # Apply transform
    moving_image = pyvips.Image.new_from_array(moving_image)
    moving_image_warped = moving_image.mapim(
        index_map, 
        interpolate=pyvips.Interpolate.new('bicubic'), 
        background=[255, 255, 255]
    ).numpy().astype(np.uint8)

    moving_mask = pyvips.Image.new_from_array(moving_mask)
    moving_mask_warped = moving_mask.mapim(
        index_map,
        interpolate=pyvips.Interpolate.new('nearest'),
        background=[0, 0, 0]
    ).numpy().astype(np.uint8)

    # Multiply image by mask to get rid of black borders
    moving_image_warped[moving_mask_warped < np.max(moving_mask_warped)] = 255

    return moving_image_warped, moving_mask_warped


def apply_deformable_transform_fullres(image: pyvips.Image, mask: pyvips.Image, grid: Any, scaling: int) -> tuple([pyvips.Image, pyvips.Image]):
    """
    Apply thin plate splines transform to the full resolution.
    """

    # Convert torch grid to pyivps grid
    dx = grid.cpu().numpy()[0, :, :, 0]
    dx = ((dx + 1) / 2) * (image.width - 1)

    dy = grid.cpu().numpy()[0, :, :, 1] 
    dy = ((dy + 1) / 2) * (image.height - 1)

    # Scale to full resolution
    dx = pyvips.Image.new_from_array(dx).resize(scaling)
    dy = pyvips.Image.new_from_array(dy).resize(scaling)

    # Ensure deformation field is exactly as large as the image
    width, height = image.width, image.height
    if (dx.width != width) or (dy.height != height):
        dx = dx.gravity("centre", width, height)
        dy = dy.gravity("centre", width, height)

    index_map = dx.bandjoin([dy])

    # Apply to image
    image_warped = image.mapim(
        index_map, 
        interpolate=pyvips.Interpolate.new('bicubic'), 
        background=[255, 255, 255]
    )
    mask_warped = mask.mapim(
        index_map,
        interpolate=pyvips.Interpolate.new('nearest'),
        background=[0, 0, 0]
    )

    return image_warped, mask_warped