import torch
import numpy as np
import pathlib
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Any, Tuple
from skimage.measure import marching_cubes
import plotly.graph_objects as go

from evaluation import compute_dice


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


def plot_ellipses(images: List[np.ndarray], ellipses: List[Tuple], centerpoints: List[Tuple], rotations: np.ndarray, save_dir: pathlib.Path) -> None:
    """
    Function to plot the fitted ellipses on the whole mounts.
    """

    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    for image, rotation, ellipse, center, ax in zip(images, rotations, ellipses, centerpoints, axs):
        
        # Create ellipse patch in matplotlib
        axes = ellipse[1]
        ellipse_patch = patches.Ellipse(center, width=axes[1], height=axes[0], angle=rotation-90, edgecolor='g', facecolor='none')
        
        # Show image, centerpoint and ellipse
        ax.imshow(image)
        ax.scatter(center[0], center[1], c="r")
        ax.add_patch(ellipse_patch)
        ax.axis("off")

    plt.savefig(save_dir.joinpath(f"ellipses.png"), dpi=300, bbox_inches="tight")
    plt.close()

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
    plt.savefig(save_dir.joinpath("02_prealignment.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_keypoint_pairs(ref_image: np.ndarray, moving_image: np.ndarray, ref_points: List, moving_points: List, matches: List, ransac_matches: List, savepath: pathlib.Path) -> None:
    """
    Function to plot the keypoint pairs on two images.
    """

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
        ref_points,
        moving_image,
        moving_points,
        ransac_matches,
        None,
        matchColor=(0,255,0),
        matchesMask=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize = (8, 8))
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

def plot_final_reconstruction(final_images: List, save_dir: pathlib.Path, tform: str) -> None:
    """
    Plot final reconstruction using the affine transformation computed from the detected keypoints.
    """

    # Overview figure of the slices
    savepath = save_dir.joinpath(f"03_final_reconstruction_{tform}.png")
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

def plot_tps_grid(points_moving: np.ndarray, points_ref: np.ndarray, tps: Any, save_path: pathlib.Path):
    """
    Function to visualize the deformation field
    obtained with the TPS.
    """

    # Create initial grid
    grid_size = 50
    min_x, min_y = np.min(points_ref, axis=0)
    max_x, max_y = np.max(points_ref, axis=0)

    x = np.linspace(min_x, max_x, grid_size)
    y = np.linspace(min_y, max_y, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack((X.flatten(), Y.flatten())).T

    # Get warped grid points
    _, warped_grid_points = tps.applyTransformation(grid_points.reshape(1, -1, 2))
    X_warped_grid = np.squeeze(warped_grid_points)[:, 0]
    Y_warped_grid = np.squeeze(warped_grid_points)[:, 1]

    _, warped_moving_points = tps.applyTransformation(points_moving.reshape(1, -1, 2))
    X_warped_moving = np.squeeze(warped_moving_points)[:, 0]
    Y_warped_moving = np.squeeze(warped_moving_points)[:, 1]

    # Source grid
    plt.subplot(1, 2, 1)
    plt.title("Source Grid")
    plt.scatter(X, Y, marker='.', color='black')
    plt.scatter(points_ref[:, 0], points_ref[:, 1], marker='o', color='blue')
    plt.scatter(points_moving[:, 0], points_moving[:, 1], marker='o', color='red') 
    # plt.xlim(min_x, max_x)
    # plt.ylim(min_y, max_y)
    plt.gca().invert_yaxis()  

    # Warped grid
    plt.subplot(1, 2, 2)
    plt.title("Warped Grid")
    plt.scatter(X, Y, marker='.', color='black')
    # plt.scatter(X_warped_grid, Y_warped_grid, marker='.', color='black')
    plt.scatter(points_ref[:, 0], points_ref[:, 1], marker='o', color='blue') 
    plt.scatter(X_warped_moving+1, Y_warped_moving+1, marker='o', color='red') 
    # plt.xlim(min_x, max_x)
    # plt.ylim(min_y, max_y)
    plt.gca().invert_yaxis()  

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return


def plot_3d_volume(volume: np.ndarray, save_dir: pathlib.Path) -> None:
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

    fig.write_image(save_dir.joinpath("3d_reconstruction.png"), engine="kaleido")
    fig.show()

    return