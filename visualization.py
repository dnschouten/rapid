import torch
import numpy as np
import pathlib
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Any, Tuple


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

def plot_prealignment(images: List[np.ndarray], contours: List[np.ndarray], save_dir: pathlib.Path) -> None:
    """
    Function to plot the images after rotation and translation adjustment.
    """

    plt.figure(figsize=(10, 5))
    for i, (image, contour) in enumerate(zip(images, contours), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image)
        plt.scatter(contour[:, 0], contour[:, 1], c="r", s=2)
        plt.axis("off")
    plt.savefig(save_dir.joinpath("02_prealignment.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_keypoint_pairs_lightglue(ref_image: np.ndarray, moving_image: np.ndarray, ref_points: torch.Tensor, moving_points: torch.Tensor, savepath: pathlib.Path) -> None:
    """
    Function to plot the lightglue keypoint pairs on two images.
    """

    # Transform keypoints to cpu
    ref_points = ref_points.cpu().numpy()
    moving_points = moving_points.cpu().numpy()
    
    images = [ref_image, moving_image]
    keypoints = [ref_points, moving_points] 

    # Plot images and keypoints
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    for c, (im, kp) in enumerate(zip(images, keypoints)):
        ax[c].imshow(im)
        ax[c].set_axis_off()
        for spine in ax[c].spines.values(): 
            spine.set_visible(False)
        ax[c].scatter(kp[:, 0], kp[:, 1], c="lime", s=4, linewidths=0)
    
    # Plot matches between keypoint pairs
    ax0, ax1 = fig.axes
    for i in range(len(ref_points)):
        line = patches.ConnectionPatch(
            xyA=(ref_points[i, 0], ref_points[i, 1]),
            xyB=(moving_points[i, 0], moving_points[i, 1]),
            coordsA=ax0.transData,
            coordsB=ax1.transData,
            axesA=ax0,
            axesB=ax1,
            zorder=1,
            color="lime",
            linewidth=0.5,
            clip_on=True,
            picker=5.0,
        )
        line.set_annotation_clip(True)
        fig.add_artist(line)
    
    # Freeze axes
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    fig.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
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



def plot_warped_images(ref_image: np.ndarray, ref_mask: np.ndarray, moving_image: np.ndarray, moving_image_warped: np.ndarray, moving_mask_warped: np.ndarray, overlap: float, savepath: pathlib.Path) -> None:
    """
    Plot the warped moving image and the overlap with the reference image.
    """

    # Get contours for visualization purposes
    cnt_moving, _ = cv2.findContours(moving_mask_warped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt_moving = np.squeeze(max(cnt_moving, key=cv2.contourArea))
    cnt_ref, _ = cv2.findContours(ref_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt_ref = np.squeeze(max(cnt_ref, key=cv2.contourArea))

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

def plot_final_reconstruction(final_reconstruction: np.ndarray, final_contours: List[List], image_paths: List[pathlib.Path], save_dir: pathlib.Path) -> None:
    """
    Plot final reconstruction using the affine transformation computed from the detected keypoints.
    """

    # Overview figure of the slices
    savepath = save_dir.joinpath("03_final_reconstruction.png")
    plt.figure(figsize=(10, 5))
    for i in range(final_reconstruction.shape[-1]):
        plt.subplot(1, final_reconstruction.shape[-1], i+1)
        plt.imshow(final_reconstruction[:, :, :, i])
        plt.axis("off")
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    # Overview figure of the overlapping contours
    savepath = save_dir.joinpath("03_final_reconstruction_contours.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(np.zeros((final_reconstruction.shape[:2])), cmap="gray")
    for cnt in final_contours:
        plt.scatter(cnt[:, 0], cnt[:, 1])
    plt.axis("off")
    plt.legend([i.stem for i in image_paths])
    plt.savefig(savepath)
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

