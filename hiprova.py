import pyvips
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import multiresolutionimageinterface as mir
import trimesh
import open3d 
import copy
import math
import torch 

from scipy import ndimage
from pathlib import Path

from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd


class Hiprova:


    def __init__(self, data_dir: Path, save_dir: Path) -> None:
        
        # Set data+save directories
        self.data_dir = data_dir
        self.save_dir = save_dir

        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # For now only support mrxs
        self.slides = self.data_dir.glob("*.mrxs")
        self.slides = sorted(list(self.slides))

        # Set level at which to load the image
        self.level = 9

        # Superglue variables
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  
        self.matcher = LightGlue(features='superpoint').eval().cuda()

        return
    
    def load_images(self) -> None:
        """
        Method to load images using pyvips.
        """

        self.images = []
        self.images_hsv = []
        
        for slide in self.slides:

            # Load image
            image = pyvips.Image.new_from_file(str(slide), level=self.level)
            image = image.flatten().numpy()
            image = ((image/np.max(image))*255).astype(np.uint8)

            # Get hsv version 
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            self.images.append(image)
            self.images_hsv.append(image_hsv)

        return 

    def load_masks(self) -> None:
        """
        Method to convert the images to masks using thresholding based on V-channel in HSV images.
        """

        self.masks = []
        vmin = 50
        vmax = 250

        for image_hsv in self.images_hsv:

            # Threshold the v-channel for a range of values
            mask = (vmin < image_hsv[:, :, 2]) * (image_hsv[:, :, 2] < vmax)
            mask = ndimage.binary_fill_holes(mask).astype("uint8")

            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Only keep the largest component in the mask
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = ((labels == largest_label)*255).astype("uint8")
            
            self.masks.append(mask)

        return

    def get_contours(self) -> None:
        """
        Method to get the centerpoint of the masks.
        """

        self.centerpoints = []
        self.contours = []
        self.contours_hull = []

        for i, mask in enumerate(self.masks):

            # Get contour
            contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = np.squeeze(max(contour, key=cv2.contourArea))

            # Compute moments of contour to get centerpoint
            moments = cv2.moments(contour)
            centerpoint = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

            # Get the convex hull of the contour
            contour_hull = cv2.convexHull(contour)
            contour_hull = np.squeeze(contour_hull)

            self.centerpoints.append(centerpoint)
            self.contours.append(contour)
            self.contours_hull.append(contour_hull)

        return
    
    def get_rdp_contour(self) -> None:
        """
        Method to get the simplified version of the contour using the RDP algorithm.
        """

        self.contours_rdp = []
        self.contours_rdp_itp = [] 

        for contour, mask in zip(self.contours, self.masks):

            # Use RDP to get approximation of contour
            contour_rdp = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
            contour_rdp = np.squeeze(contour_rdp)

            # Interpolate the RDP contour. We do this by drawing the contour on an 
            # empty frame and converting it back to a contour. 
            rdp_mask = np.zeros_like(mask)
            rdp_mask = cv2.drawContours(rdp_mask, [contour_rdp], -1, 255, thickness=cv2.FILLED)
            contour_rdp_itp, _ = cv2.findContours(rdp_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour_rdp_itp = np.squeeze(max(contour_rdp_itp, key=cv2.contourArea))

            self.contours_rdp.append(contour_rdp)
            self.contours_rdp_itp.append(contour_rdp_itp)

        return

    def get_rotations(self) -> None:
        """
        Method to get the rotation of the prostate based on the 
        eigenvectors of the contour of the mask.
        """

        self.rotations = []
        self.eigenvectors = []

        for i, (contour, image) in enumerate(zip(self.contours_hull, self.images)):
            
            # Get eigenvectors and angle
            _, eigenvector = cv2.PCACompute(np.squeeze(contour).reshape(-1, 2).astype("uint16"), mean=np.empty((0)))
            angle = np.degrees(np.arctan2(eigenvector[1, 1], eigenvector[1, 0]))

            self.rotations.append(np.round(angle, 2))
            self.eigenvectors.append(eigenvector)  

            # Plot the contour and eigenvectors
        plt.figure(figsize=(20, 10))
        for c, (image, contour, eigenvector) in enumerate(zip(self.images, self.contours_hull, self.eigenvectors), 1):
            plt.subplot(1, len(self.images), c)
            plt.imshow(image)
            plt.plot(contour[:, 0], contour[:, 1], c="r")
            plt.quiver(*contour.mean(axis=0), *eigenvector[0, :], color="g", scale=10)
            plt.quiver(*contour.mean(axis=0), *eigenvector[1, :], color="b", scale=10)
            plt.axis("off")
        plt.savefig(self.save_dir.joinpath(f"eigenvectors.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return


    def match_centerpoints(self) -> None:
        """
        Function to match all images based on the centerpoints.
        """

        self.translated_images = []
        self.translated_contours = []

        # Get centerpoint of first image
        self.common_cp = np.array(self.images[0].shape[:-1])
        self.common_cp = (0.5*self.common_cp).astype("int")

        for contour, center, image in zip(self.contours, self.centerpoints, self.images):

            # Get displacement
            displacement = self.common_cp - center

            # Translate the contour and image
            translated_contour = contour + displacement
            translation_matrix = np.float32([[1, 0, displacement[0]], [0, 1, displacement[1]]])
            translated_image = cv2.warpAffine(image, translation_matrix, image.shape[:-1][::-1], borderValue=(255, 255, 255))
            
            self.translated_contours.append(translated_contour)
            self.translated_images.append(translated_image)

        return

    def match_rotations(self) -> None:
        """
        Function to match all images based on the rotation.
        """

        self.rotated_images = []
        self.rotated_contours = []

        for contour, rotation, image in zip(self.translated_contours, self.rotations, self.translated_images):

            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(tuple(self.common_cp.astype("int16")), rotation, 1)

            # Rotate the contour and image
            rotated_contour = cv2.transform(np.expand_dims(contour, axis=0), rotation_matrix)
            rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[:-1][::-1], borderValue=(255, 255, 255))

            self.rotated_contours.append(rotated_contour)
            self.rotated_images.append(rotated_image)

        # Get non zero areas of contour and crop images based on these indices for efficiency
        pad = 0.05
        self.xmin = np.min([np.min(contour[:, :, 0]) for contour in self.rotated_contours])
        self.xmax = np.max([np.max(contour[:, :, 0]) for contour in self.rotated_contours])
        self.xmin = np.max([0, self.xmin-(pad*(self.xmax-self.xmin))])
        self.xmax = np.min([self.rotated_images[0].shape[1], self.xmax+(pad*(self.xmax-self.xmin))])
        self.ymin = np.min([np.min(contour[:, :, 1]) for contour in self.rotated_contours])
        self.ymax = np.max([np.max(contour[:, :, 1]) for contour in self.rotated_contours])
        self.ymin = np.max([0, self.ymin-(pad*(self.ymax-self.ymin))])
        self.ymax = np.min([self.rotated_images[0].shape[0], self.ymax+(pad*(self.ymax-self.ymin))])

        self.rotated_images = [image[int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax)] for image in self.rotated_images]    
        self.rotated_contours = [contour - np.array([self.xmin, self.ymin]) for contour in self.rotated_contours]

        # Stack all rotated images into a 3D volume
        self.initial_reconstruction = np.stack(self.rotated_images, axis=-1)

        # Plot initial reconstruction
        plt.figure(figsize=(20, 10))
        for i in range(self.initial_reconstruction.shape[-1]):
            plt.subplot(1, self.initial_reconstruction.shape[-1], i+1)
            plt.imshow(self.initial_reconstruction[:, :, :, i])
            plt.axis("off")
        plt.savefig(self.save_dir.joinpath("initial_reconstruction.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return

    def finetune_reconstruction_icp(self) -> None:
        """
        Function to finetune the match between adjacent images using the 
        contour of adjacent images.
        """

        # We use the mid slice as reference point and move all images toward this slice.
        mid_slice = int(np.ceil(len(self.rotated_images)//2))
        self.finetuned_images = [None] * mid_slice + [self.rotated_images[mid_slice]] + [None] * (len(self.rotated_images)-mid_slice-1)
        self.finetuned_contours = [None] * mid_slice + [self.rotated_contours[mid_slice]] + [None] * (len(self.rotated_contours)-mid_slice-1)

        self.moving_indices = list(np.arange(0, mid_slice)[::-1]) + list(np.arange(mid_slice+1, len(self.rotated_images)))
        self.ref_indices = list(np.arange(0, mid_slice)[::-1] + 1) + list(np.arange(mid_slice+1, len(self.rotated_images)) - 1)

        # Iteratively perform ICP registration for adjacent pairs and update the reference points.
        for mov, ref in zip(self.moving_indices, self.ref_indices):

            # Get reference image and contour. This serves as the fixed image
            ref_contour = np.squeeze(self.finetuned_contours[ref])
            ref_contour = np.hstack([ref_contour, np.ones((ref_contour.shape[0], 1))])

            # Get the current contour and image, this will be moved towards the fixed image
            moving_contour = np.squeeze(self.rotated_contours[mov])
            moving_contour = np.hstack([moving_contour, np.ones((moving_contour.shape[0], 1))])
            moving_image = self.rotated_images[mov]

            # Perform ICP registration with trimesh
            m, _, _ = trimesh.registration.icp(
                moving_contour, 
                ref_contour, 
                initial=None, 
                max_iterations=1000
            )

            # Get translation and rotation
            translation = m[:2, 3]
            sx = np.linalg.norm(m[:3, 0])
            sy = np.linalg.norm(m[:3, 1])
            sz = np.linalg.norm(m[:3, 2])
            m_rot = m[:3, :3] / np.array([sx, sy, sz])
            rot = math.degrees(np.arctan2(m_rot[1, 0], m_rot[0, 0]))

            # Compute moments of src contour to get centerpoint to warp around
            moments = cv2.moments(moving_contour)
            src_centerpoint = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

            # Create transformation matrix from translation and rotation. We use the 
            # extracted translation and rotation rather than the full matrix
            # to avoid the scaling component of the transform.
            m_tform = cv2.getRotationMatrix2D(tuple(src_centerpoint), rot, 1)
            m_tform[:2, 2] += translation

            # Apply transformation to image and contour
            moving_image = cv2.warpAffine(moving_image, m_tform[:2], moving_image.shape[:-1][::-1], borderValue=(255, 255, 255))
            moving_contour = cv2.transform(np.expand_dims(moving_contour, axis=0), m_tform)

            # Save finetuned image and contour in list
            self.finetuned_images[mov] = moving_image
            self.finetuned_contours[mov] = moving_contour

        # Stack all finetuned images into a 3D volume
        self.icp_reconstruction = np.stack(self.finetuned_images, axis=-1)

        # Save ICP reconstruction
        plt.figure(figsize=(20, 10))
        for i in range(self.icp_reconstruction.shape[-1]):
            plt.subplot(1, self.icp_reconstruction.shape[-1], i+1)
            plt.imshow(self.icp_reconstruction[:, :, :, i])
            plt.axis("off")
        plt.savefig(self.save_dir.joinpath("icp_reconstruction.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return
    
    def finetune_reconstruction_lightglue(self) -> None:
        """
        Function to finetune the match between adjacent images using lightglue.
        """

        # We use the mid slice as reference point and move all images toward this slice.
        mid_slice = int(np.ceil(len(self.rotated_images)//2))
        self.final_images = [None] * mid_slice + [self.rotated_images[mid_slice]] + [None] * (len(self.rotated_images)-mid_slice-1)
        self.final_contours = [None] * mid_slice + [self.rotated_contours[mid_slice]] + [None] * (len(self.rotated_contours)-mid_slice-1)

        self.moving_indices = list(np.arange(0, mid_slice)[::-1]) + list(np.arange(mid_slice+1, len(self.rotated_images)))
        self.ref_indices = list(np.arange(0, mid_slice)[::-1] + 1) + list(np.arange(mid_slice+1, len(self.rotated_images)) - 1)

        # Iteratively perform lightglue matching for adjacent pairs and update the reference points.
        for mov, ref in zip(self.moving_indices, self.ref_indices):

            # Get reference image and contour. This serves as the fixed image
            ref_image = self.final_images[ref]

            # Prepare for lightglue
            ref_image = torch.tensor(ref_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()

            # Get the current contour and image, this will be moved towards the fixed image
            moving_image = self.rotated_images[mov]

            # Prepare for lightglue
            moving_image = torch.tensor(moving_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()

            # Extract features
            ref_features = self.extractor.extract(ref_image)
            moving_features = self.extractor.extract(moving_image)

            # Match features
            matches01 = self.matcher({'image0': ref_features, 'image1': moving_features})
            ref_features, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
            matches = matches01['matches']
            points_ref = ref_features['keypoints'][matches[..., 0]]
            points_moving = moving_features['keypoints'][matches[..., 1]]

            # Save image of corresponding matches
            axes = viz2d.plot_images([ref_image, moving_image])
            viz2d.plot_matches(points_ref, points_moving, color='lime', lw=0.2)
            viz2d.save_plot(self.save_dir.joinpath(f"keypoints_{mov}_to_{ref}.png"))

            # Convert back to numpy
            points_ref = points_ref.cpu().numpy()
            points_moving = points_moving.cpu().numpy()
            ref_image = ref_image.cpu().numpy().transpose((1, 2, 0))
            moving_image = moving_image.cpu().numpy().transpose((1, 2, 0))

            # Find and apply transformation
            tform = "partial_affine"
            if tform == "full_affine":
                rows, cols, _ = moving_image.shape
                affine_matrix, inliers = cv2.estimateAffine2D(points_moving, points_ref)            
                moving_image_warped = cv2.warpAffine(
                    (moving_image * 255).astype("uint8"), 
                    affine_matrix,  
                    (cols, rows), 
                    borderValue=(255, 255, 255)
                )
            elif tform == "partial_affine":
                # Compute centroids
                centroid_fixed = np.mean(points_ref, axis=0)
                centroid_moving = np.mean(points_moving, axis=0)

                # Shift the keypoints so that both sets have a centroid at the origin
                points_ref_centered = points_ref - centroid_fixed
                points_moving_centered = points_moving - centroid_moving

                # Compute the rotation matrix
                H = np.dot(points_moving_centered.T, points_ref_centered)
                U, _, Vt = np.linalg.svd(H)
                R = np.dot(Vt.T, U.T)

                # Create the combined rotation and translation matrix
                affine_matrix = np.zeros((2, 3))
                affine_matrix[:2, :2] = R
                affine_matrix[:2, 2] = centroid_fixed - np.dot(R, centroid_moving)

                # Apply transformation
                rows, cols, _ = moving_image.shape
                moving_image_warped = cv2.warpAffine(
                        (moving_image * 255).astype("uint8"), 
                        affine_matrix,  
                        (cols, rows), 
                        borderValue=(255, 255, 255)
                )

            # Save final image and contours
            self.final_images[mov] = moving_image_warped.astype("uint8")

            # Save image of result 
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(ref_image)
            plt.title("Reference image")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(moving_image)
            plt.title("Moving image")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(moving_image_warped)
            plt.title("Moving image warped")
            plt.axis("off")
            plt.savefig(self.save_dir.joinpath(f"warped_{mov}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        self.final_reconstruction = np.stack(self.final_images, axis=-1)

        # Save final reconstruction
        plt.figure(figsize=(20, 10))
        for i in range(self.final_reconstruction.shape[-1]):
            plt.subplot(1, self.final_reconstruction.shape[-1], i+1)
            plt.imshow(self.final_reconstruction[:, :, :, i])
            plt.axis("off")
        plt.savefig(self.save_dir.joinpath("final_reconstruction.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return

    
    def plot_3d_volume(self):
        """
        Function to plot all the levels of the 3D reconstructed volume in a single 3D plot.
        """
    
        # Plot the 3D volume in a single plot
        

        return
    
    # q: I want to plot an image in VS code in debug mode using matplotlib, how do I do this?
    # a: https://stackoverflow.com/questions/19410042/how-to-make-ipython-notebook-matplotlib-plot-inline