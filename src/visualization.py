import numpy as np
import pathlib
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Any, Tuple
from skimage.measure import marching_cubes
import plotly.graph_objects as go

from evaluation import compute_dice
from transforms import *
from utils import *


def plot_initial_reconstruction(images: List[np.ndarray], save_dir: pathlib.Path) -> None:
    """
    Function to plot the initial situation.
    """
    
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(images, 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image)
        plt.axis("off")
    plt.savefig(save_dir.joinpath("01_initial_situation.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_ellipses(images: List[np.ndarray], ellipses: List[Tuple], ref_idx: int, save_dir: pathlib.Path) -> None:
    """
    Function to plot the fitted ellipses on the whole mounts.
    """

    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    for c, (image, ellipse, ax) in enumerate(zip(images, ellipses, axs)):
        
        # Create ellipse patch in matplotlib
        center, axes, rotation = ellipse
        ellipse_patch = patches.Ellipse(center, width=axes[1], height=axes[0], angle=rotation-90, edgecolor='g', facecolor='none')
        
        # Show image, centerpoint and ellipse
        ax.imshow(image)
        ax.scatter(center[0], center[1], c="r")
        ax.add_patch(ellipse_patch)
        ax.axis("off")
        if c == ref_idx:
            ax.set_title("ref")
        else:
            ax.set_title(f"moving")

    plt.savefig(save_dir.joinpath(f"02_ellipses.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_stain_normalization(images: List[np.ndarray], normalized_images: List[np.ndarray], savepath: pathlib.Path) -> None:
    """
    Function to plot the results of the stain normalization procedure.
    """

    plt.figure(figsize=(10, 5))
    plt.suptitle("Macenko stain normalization effect")
    for c, (image, normalized_image) in enumerate(zip(images, normalized_images), 1):
        plt.subplot(2, len(images), c)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(2, len(images), c+len(images))
        plt.imshow(normalized_image)
        plt.axis("off")
    plt.savefig(savepath, dpi=300, bbox_inches="tight")

    return


def plot_prealignment(images: List[np.ndarray], save_dir: pathlib.Path) -> None:
    """
    Function to plot the images after rotation and translation adjustment.
    """

    plt.figure(figsize=(10, 5))
    for c, image in enumerate(images, 1):
        plt.subplot(1, len(images), c)
        plt.imshow(image)
        plt.axis("off")
    plt.savefig(save_dir.joinpath("03_prealignment.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_keypoint_pairs(ref_image: np.ndarray, moving_image: np.ndarray, ref_points: List, moving_points: List, tform: str, savepath: pathlib.Path) -> None:
    """
    Function to plot the keypoint pairs on two images.
    """

    # Compute ransac matches to visualize the effect of RANSAC
    if tform == "affine":
        ref_points_ransac, moving_points_ransac = apply_affine_ransac(
            moving_points = moving_points,
            ref_points = ref_points,
            image = moving_image
        )
    elif tform == "deformable":
        ref_points_ransac, moving_points_ransac = apply_deformable_ransac(
            moving_points = moving_points,
            ref_points = ref_points,
            device = "cuda"
        )

    # Define matches and keypoints according to opencv standards
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(ref_points))]
    ransac_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(ref_points_ransac))]
    
    ref_points = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in ref_points]
    moving_points = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in moving_points]
    ref_points_ransac = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in ref_points_ransac]
    moving_points_ransac = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in moving_points_ransac]

    # Draw matches on canvas
    result = cv2.drawMatches(
        ref_image, 
        ref_points, 
        moving_image, 
        moving_points, 
        matches, 
        None, 
        matchColor=(0,255,0), 
        matchesMask=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    result_ransac = cv2.drawMatches(
        ref_image,
        ref_points_ransac,
        moving_image,
        moving_points_ransac,
        ransac_matches,
        None,
        matchColor=(0,255,0),
        matchesMask=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize = (6, 6))
    plt.subplot(121)
    plt.imshow(result)
    plt.title(f"original matches: (n={len(matches)})")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(result_ransac)
    plt.title(f"RANSAC matches: (n={len(ransac_matches)})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    return

def plot_warped_images(ref_image: np.ndarray, ref_mask: np.ndarray, moving_image: np.ndarray, moving_image_warped: np.ndarray, moving_mask_warped: np.ndarray, savepath: pathlib.Path) -> None:
    """
    Plot the warped moving image and the overlap with the reference image.
    """

    # Get contours for visualization purposes
    cnt_moving, _ = cv2.findContours(moving_mask_warped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt_moving = np.squeeze(max(cnt_moving, key=cv2.contourArea))
    cnt_ref, _ = cv2.findContours(ref_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt_ref = np.squeeze(max(cnt_ref, key=cv2.contourArea))

    # Compute overlap
    overlap = compute_dice(moving_mask_warped, ref_mask)

    # Save image of result 
    plt.figure(figsize=(10, 5))
    plt.subplot(141)
    plt.imshow(ref_image)
    plt.title("Reference image")
    plt.axis("off")
    plt.subplot(142)
    plt.imshow(moving_image)
    plt.title("Moving image")
    plt.axis("off")
    plt.subplot(143)
    plt.imshow(moving_image_warped)
    plt.title("Moving image warped")
    plt.axis("off")
    plt.subplot(144)
    plt.imshow(np.zeros_like(moving_image_warped))
    plt.scatter(cnt_ref[:, 0], cnt_ref[:, 1], c="r", s=2)
    plt.scatter(cnt_moving[:, 0], cnt_moving[:, 1], c="b", s=2)
    plt.title(f"dice: {overlap:.2f}")
    plt.axis("off")
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_warped_deformable_images(ref_image: np.ndarray, moving_image: np.ndarray, moving_image_warped: np.ndarray, grid: Any, savepath: pathlib.Path) -> None:
    """
    Function to show the deformable warped image and the corresponding grid.
    """

    # Apply deformable registration to grid for visualization purposes
    image_shape = moving_image.shape[:2]
    warped_grid = grid_to_image(image_size=image_shape, grid=grid)

    plt.figure()
    plt.subplot(141)
    plt.imshow(ref_image)
    plt.axis("off")
    plt.title("ref")
    plt.subplot(142)
    plt.imshow(moving_image)
    plt.axis("off")
    plt.title("mov")
    plt.subplot(143)
    plt.imshow(warped_grid)
    plt.axis("off")
    plt.title("deformable grid")
    plt.subplot(144)
    plt.imshow(moving_image_warped)
    plt.axis("off")
    plt.title("warped")
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_final_reconstruction(final_images: List, save_dir: pathlib.Path, tform: str) -> None:
    """
    Plot final reconstruction using the affine transformation computed from the detected keypoints.
    """

    # Get path to save
    image_indices = sorted([i.name.split("_")[0] for i in save_dir.glob("*.png")])
    idx = int(image_indices[-1]) + 1
    savepath = save_dir.joinpath(f"{str(idx).zfill(2)}_final_reconstruction_{tform}.png")

    plt.figure(figsize=(10, 5))
    for c, im in enumerate(final_images):
        plt.subplot(1, len(final_images), c+1)
        plt.imshow(im)
        plt.axis("off")
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()
    
    return

def plot_interpolated_contour(slice_a: np.ndarray, contour_a: List[int], slice_b: np.ndarray, contour_b: List[int], slice_c: np.ndarray, contour_c: List[int], save_path: pathlib.Path) -> None: 
    """
    Function to plot the interpolated contour between
    adjacent slides. Serves as a sanity check for the
    interpolation method
    """

    plt.figure()
    plt.subplot(131)
    plt.imshow(slice_a, cmap="gray")
    plt.scatter(contour_a[:, 0], contour_a[:, 1], c="r", s=1)
    plt.title("previous")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(slice_b, cmap="gray")
    plt.scatter(contour_b[:, 0], contour_b[:, 1], c="r", s=1)
    plt.title(f"interpolated")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(slice_c, cmap="gray")
    plt.scatter(contour_c[:, 0], contour_c[:, 1], c="r", s=1)
    plt.title("next")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_tre_per_pair(ref_image: np.ndarray, moving_image: np.ndarray, ref_points: np.ndarray, moving_points: np.ndarray, tre: float, savepath: pathlib.Path) -> None:
    """
    Function to visualize the TRE per pair of images.
    """

    plt.figure(figsize=(8, 4))
    plt.suptitle(f"TRE: {tre:.2f} microns (n={len(ref_points)})")   
    plt.subplot(131)
    plt.imshow(ref_image)
    plt.scatter(ref_points[:, 0], ref_points[:, 1], c="r", s=5)
    plt.title("im1")
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(moving_image)
    plt.scatter(moving_points[:, 0], moving_points[:, 1], c="b", s=5)
    plt.title("im2")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(np.zeros_like(ref_image))
    plt.scatter(ref_points[:, 0], ref_points[:, 1], c="r", s=5)
    plt.scatter(moving_points[:, 0], moving_points[:, 1], c="b", s=5)
    for i in range(len(ref_points)):
        plt.plot([ref_points[i, 0], moving_points[i, 0]], [ref_points[i, 1], moving_points[i, 1]], c="w", lw=1)
    plt.legend(["ref", "moving"])
    plt.axis("off")
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_3d_volume(volume: np.ndarray, savepath: pathlib.Path) -> None:
    """
    Method to plot all the levels of the 3D reconstructed volume in a single 3D plot.
    """

    # Extract surface mesh from 3D volume
    verts, faces, _, _ = marching_cubes(volume)

    # Plot using plotly
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.5,
            color='pink'
        )
    ])

    fig.update_layout(scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
        aspectmode="data"),
        margin=dict(t=0, b=0, l=0, r=0)
    )

    fig.write_image(savepath, engine="kaleido")
    fig.show()

    return
