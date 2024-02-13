import cv2
import numpy as np
import torch
import SimpleITK as sitk
import pathlib

from matplotlib import pyplot as plt
from typing import Tuple, List, Any
from scipy.interpolate import interpn, interp1d
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from torchvision.transforms import ToTensor, ToPILImage
from radiomics import shape

from transforms import apply_affine_transform


def find_dorsal_rotation(mask: np.ndarray, ellipse: Any, center: np.ndarray, savepath: pathlib.Path) -> int:
    """
    Function to find the flat dorsal side of the prostate by computing
    the smallest bounding box around the prostate and then checking the 
    distance from bounding box to the prostate contour. The two box points
    with the smallest distance to the contour are then considered
    the dorsal side.

    Returns orientation needed to rotate the prostate on dorsal side.
    """

    # Find regular rotation based on ellipse
    center, axis, rotation = ellipse
    if axis[1] > axis[0]:
        rotation += 90

    # Apply rotation 
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), rotation, 1)
    _, mask = apply_affine_transform(
        image = np.zeros((mask.shape[0], mask.shape[1], 3)),
        tform = rotation_matrix,
        mask = mask,
    )

    # Find contour
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.squeeze(max(contour, key=cv2.contourArea))

    # Find bbox corners
    bbox = cv2.minAreaRect(contour)
    bbox_corners = cv2.boxPoints(bbox)

    upper_bbox_corners = bbox_corners[bbox_corners[:, 1] < np.mean(bbox_corners[:, 1])]
    lower_bbox_corners = bbox_corners[bbox_corners[:, 1] > np.mean(bbox_corners[:, 1])]
    assert upper_bbox_corners.shape == lower_bbox_corners.shape, "Upper and lower bbox corners must have the same shape."

    # Compute distance from bbox corners to contour
    upper_corner_dist = np.min(np.min(cdist(upper_bbox_corners, contour), axis=-1))
    lower_corner_dist = np.min(np.min(cdist(lower_bbox_corners, contour), axis=-1))

    rotation = 180 if lower_corner_dist > upper_corner_dist else 0
    colours = ["g", "r"] if rotation == 180 else ["r", "g"]

    # Sanity check plot
    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.plot(contour[:, 0], contour[:, 1], "b")
    plt.scatter(upper_bbox_corners[:, 0], upper_bbox_corners[:, 1], c=colours[0])
    plt.scatter(lower_bbox_corners[:, 0], lower_bbox_corners[:, 1], c=colours[1])
    plt.title(f"Rotate {rotation} degrees.")
    plt.savefig(savepath)
    plt.close()
    
    return rotation


def compute_line_intersection(line1: np.ndarray, line2: np.ndarray) -> Tuple[float, float]:
    """
    Function to compute the intersection between two lines.

    Input: line1 and line2 are tuples of the form (x1, y1, x2, y2).
    """

    # Line represented as a1x + b1y = c1
    a1 = line1[1][1] - line1[0][1]
    b1 = line1[0][0] - line1[1][0]
    c1 = a1 * line1[0][0] + b1 * line1[0][1]

    # Line segment represented as a2x + b2y = c2
    a2 = line2[1][1] - line2[0][1]
    b2 = line2[0][0] - line2[1][0]
    c2 = a2 * line2[0][0] + b2 * line2[0][1]

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:  # The lines are parallel
        return None
    else:
        x = np.round((b2 * c1 - b1 * c2) / determinant, 3)
        y = np.round((a1 * c2 - a2 * c1) / determinant, 3)
        online1 = (
            (min(line1[0][0], line1[1][0]) < x < max(line1[0][0], line1[1][0]))
            or (min(line1[0][1], line1[1][1]) < y < max(line1[0][1], line1[1][1]))
        )
        online2 = (
            (min(line2[0][0], line2[1][0]) < x < max(line2[0][0], line2[1][0]))
            or (min(line2[0][1], line2[1][1]) < y < max(line2[0][1], line2[1][1]))
        )

        if online1 and online2:
            return (x, y)
        else:
            return None


def resample_contour_linear(contour: np.ndarray, num_points: int) -> np.ndarray:
    """
    Convenience function to resample a contour to a certain number of points.
    """

    assert len(contour.shape) == 2, "contour must be 2 dimensional"
    assert contour.shape[-1] == 2, "contour must be of shape Nx2"

    # Reshape and normalize the contour points
    if not np.all(contour[0, :] == contour[-1, :]):
        contour = np.vstack([contour, contour[0, :]])  

    t = np.linspace(0, 1, len(contour))

    # Interpolate x and y coordinates separately
    fx = interp1d(t, contour[:, 0], kind='linear', fill_value="extrapolate")
    fy = interp1d(t, contour[:, 1], kind='linear', fill_value="extrapolate")

    # Generate new equally spaced parameter values
    new_t = np.linspace(0, 1, num_points)

    # Calculate the new contour points
    new_contour = np.zeros((num_points, 2))
    new_contour[:, 0] = fx(new_t)
    new_contour[:, 1] = fy(new_t)

    return new_contour


def resample_contour_radial(contour: np.ndarray, num_points: int) -> np.ndarray:
    """
    Function to radially resample a contour to a certain number of points
    """

    assert len(contour.shape) == 2, "contour must be 2 dimensional"
    assert contour.shape[-1] == 2, "contour must be of shape Nx2"

    # Close contour if required
    if not np.all(contour[0, :] == contour[-1, :]):
        contour = np.vstack([contour, contour[0, :]])
    
    # Get the center point of the contour
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Get the radial lines to sample from
    angles = np.linspace(0, 2 * np.pi, num_points)
    radial_lines = []

    xmax = (np.max(contour[:, 0]) - np.min(contour[:, 0]))*2
    ymax = (np.max(contour[:, 1]) - np.min(contour[:, 1]))*2

    for angle in angles:
        end_x = cx + xmax * np.cos(angle)
        end_y = cy + ymax * np.sin(angle)
        radial_lines.append(np.array([[cx, cy], [end_x, end_y]]))

    # Perform radial resampling
    resampled_contour = []

    for rline in radial_lines:

        intersect_per_rline = []

        for idx in range(len(contour)-1):

            # Get intersection between contour and radial line
            cline = np.vstack([
                contour[idx, :],
                contour[idx+1, :]
            ])
            intersect = compute_line_intersection(rline, cline)

            # Only keep if it exists
            if intersect is not None:
                intersect_per_rline.append(intersect)

        # In case of multiple intersections take the outer one
        if len(intersect_per_rline) > 0:
            
            outer_idx = np.argmax(
                [np.sqrt((ix-cx)**2 + (iy-cy)**2) for ix, iy in intersect_per_rline]
            )
            resampled_contour.append(intersect_per_rline[outer_idx])
    
    resampled_contour = np.array(resampled_contour)

    if not len(resampled_contour) == num_points:
        resampled_contour = resample_contour_linear(resampled_contour, num_points)

    return resampled_contour


def simplify_contour(contour: np.ndarray) -> np.ndarray:
    """
    Function to simplify a contour.
    """

    assert len(contour.shape) == 2, "contour must be 2 dimensional"
    assert contour.shape[-1] == 2, "contour must be of shape Nx2"

    # Close contour if required
    if not np.all(contour[0, :] == contour[-1, :]):
        contour = np.vstack([contour, contour[0, :]])

    # Simplify contour
    epsilon = 0.001 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    contour = np.squeeze(contour)

    # Close contour if required
    if not np.all(contour[0, :] == contour[-1, :]):
        contour = np.vstack([contour, contour[0, :]])

    return contour


def grid_to_image(image_size: Tuple, grid: torch.Tensor) -> np.ndarray:
    """
    Draw a grid on a blank image.
    """

    # Create blank image and define grid size
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255
    row_step, col_step = 100, 100

    # Draw horizontal and vertical lines
    line_color = (0, 0, 0)
    line_width = 1
    for i in range(0, image_size[0], row_step):
        image[i:i+line_width, :, :] = line_color

    for j in range(0, image_size[1], col_step):
        image[:, j:j+line_width, :] = line_color

    # Apply the grid
    tensor_image = ToTensor()(image).unsqueeze(0).cuda()
    warped_image = torch.nn.functional.grid_sample(tensor_image, grid, mode='bilinear', align_corners=False)

    # Remove the batch dimension and convert back to PIL image
    warped_image = warped_image.squeeze(0)
    warped_image = ToPILImage()(warped_image)

    return warped_image

