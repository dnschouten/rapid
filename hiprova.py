import pyvips
import matplotlib.pyplot as plt
import numpy as np
import cv2
import trimesh
import math
import torch 
import copy 
import SimpleITK as sitk

from scipy import ndimage
from pathlib import Path
from scipy.spatial import procrustes, distance
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import matplotlib.patches as patches
from sklearn.metrics import normalized_mutual_info_score
from skimage.metrics import structural_similarity as ssim


from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd


class Hiprova:

    def __init__(self, data_dir: Path, mask_dir: Path, save_dir: Path) -> None:
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.mask_dir = mask_dir

        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # For now only support tif
        self.image_paths = self.data_dir.glob("*.tif")
        self.image_paths = sorted(list(self.image_paths))
        self.mask_paths = [self.mask_dir.joinpath(i.name) for i in self.image_paths]

        # Set level at which to load the image
        self.image_level = 9
        self.mask_level = 6

        # Initialize Lightglue 
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  
        self.matcher = LightGlue(features='superpoint').eval().cuda()

        # Create directories
        self.save_dir.joinpath("keypoints").mkdir(parents=True, exist_ok=True)
        self.save_dir.joinpath("warps").mkdir(parents=True, exist_ok=True)

        return
    
    def load_images(self) -> None:
        """
        Method to load images using pyvips.
        """

        self.images = []
        self.images_hsv = []
        
        for im_path in self.image_paths:

            # Load image
            if im_path.suffix == ".mrxs":
                image = pyvips.Image.new_from_file(str(im_path), level=self.image_level)
            elif im_path.suffix == ".tif":
                image = pyvips.Image.new_from_file(str(im_path), page=self.image_level)
            else:
                raise ValueError("Sorry, only .tifs and .mrxs are supported.")

            # Dispose of alpha band if present
            if image.bands == 4:
                image = image.flatten().numpy()
            elif image.bands == 3: 
                image = image.numpy()

            # Normalize
            image = ((image/np.max(image))*255).astype(np.uint8)

            # Get hsv version 
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Save images
            self.images.append(image)
            self.images_hsv.append(image_hsv)

        # Plot initial reconstruction
        plt.figure(figsize=(20, 10))
        for i, image in enumerate(self.images, 1):
            plt.subplot(1, len(self.images), i)
            plt.imshow(image)
            plt.axis("off")
        plt.savefig(self.save_dir.joinpath("01_initial_situation.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return 

    def load_masks(self) -> None:
        """
        Method to load the masks. These are based on the tissue segmentation algorithm described in:
        Bándi P, Balkenhol M, van Ginneken B, van der Laak J, Litjens G. 2019. Resolution-agnostic tissue segmentation in
        whole-slide histopathology images with convolutional neural networks. PeerJ 7:e8242 DOI 10.7717/peerj.8242.
        """

        self.masks = []
        
        for mask_path in self.mask_paths:

            # Load mask
            mask = pyvips.Image.new_from_file(str(mask_path), page=self.mask_level)
            mask = mask.numpy()

            # Fill some holes 
            mask = ndimage.binary_fill_holes(mask).astype("uint8")

            # Smooth the mask a bit
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Only keep the largest component in the mask
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = ((labels == largest_label)*255).astype("uint8")

            self.masks.append(mask)

        return

    def apply_masks(self) -> None:
        """
        Method to mask the images based on the convex hull of the contour. This allows
        for some information to be retained outside of the prostate.
        """

        self.masked_images = []
        self.hulls = []
        self.contours = []

        for image, mask in zip(self.images, self.masks):
                
            # Get contour from mask
            contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = np.squeeze(max(contour, key=cv2.contourArea))

            # Get convex hull of contour
            hull = np.squeeze(cv2.convexHull(contour))
            hull = np.vstack([hull, hull[0, :]])
            
            # Create mask from convex hull
            hull_mask = np.zeros(image.shape[:-1], dtype="uint8")
            cv2.fillPoly(hull_mask, [hull.astype("int")], color=(255, 255, 255))

            # Apply mask to image
            masked_image = copy.copy(image)
            
            mask_type = "hull"
            if mask_type == "hull":
                masked_image[hull_mask == 0] = [255, 255, 255]
            elif mask_type == "regular": 
                masked_image[mask == 0] = [255, 255, 255]

            # Save masked image and convex hull
            self.contours.append(contour)
            self.hulls.append(hull)
            self.masked_images.append(masked_image)

        return

    def find_rotations(self) -> None:
        """
        Method to get the rotation of the prostate based on an
        ellipsoid approximating the fit of the prostate.
        """

        self.rotations = []
        self.centerpoints = []
        self.ellipses = []

        # Find ellipse for all images
        for contour in self.contours:
            
            # Fit ellipse based on contour 
            ellipse = cv2.fitEllipse(contour)
            center, axes, rotation = ellipse

            # Correct rotation for opencv/mpl conventions
            self.rotations.append(rotation)
            self.centerpoints.append(center)
            self.ellipses.append(ellipse)

        # Plot resulting ellipse and contour
        fig, axs = plt.subplots(1, len(self.images), figsize=(20, 10))
        for image, rotation, ellipse, center, ax in zip(self.masked_images, self.rotations, self.ellipses, self.centerpoints, axs):
            
            # Create ellipse patch in matplotlib
            axes = ellipse[1]
            ellipse_patch = patches.Ellipse(center, width=axes[1], height=axes[0], angle=rotation-90, edgecolor='g', facecolor='none')
            
            # Show image, centerpoint and ellipse
            ax.imshow(image)
            ax.scatter(center[0], center[1], c="r")
            ax.add_patch(ellipse_patch)
            ax.axis("off")

        plt.savefig(self.save_dir.joinpath(f"ellipses.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return

    def prealignment(self) -> None:
        """
        Method to match the rotations of adjacent images.
        """

        self.rotated_images = []
        self.rotated_masks = []
        self.rotated_contours = []

        # Find common centerpoint of all ellipses to orient towards
        self.common_center = np.mean([i[0] for i in self.ellipses], axis=0).astype("int")

        for image, mask, contour, rotation, center in zip(self.masked_images, self.masks, self.contours, self.rotations, self.centerpoints):

            # Adjust rotation 
            rotation_matrix = cv2.getRotationMatrix2D(tuple(center), rotation, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[:-1][::-1], borderValue=(255, 255, 255))
            rotated_mask = cv2.warpAffine(mask, rotation_matrix, mask.shape[::-1], borderValue=(0, 0, 0))
            rotated_contour = cv2.transform(np.expand_dims(contour, axis=0), rotation_matrix)
            
            # Adjust translation 
            translation = self.common_center - center
            translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
            rotated_image = cv2.warpAffine(rotated_image, translation_matrix, rotated_image.shape[:-1][::-1], borderValue=(255, 255, 255))
            rotated_mask = cv2.warpAffine(rotated_mask, translation_matrix, rotated_mask.shape[::-1], borderValue=(0, 0, 0))
            rotated_mask = ((rotated_mask > 128)*255).astype("uint8")
            rotated_contour = cv2.transform(rotated_contour, translation_matrix)

            self.rotated_images.append(rotated_image)
            self.rotated_masks.append(rotated_mask)
            self.rotated_contours.append(np.squeeze(rotated_contour))

        # Plot resulting prealignment
        plt.figure(figsize=(20, 10))
        for i, (image, contour) in enumerate(zip(self.rotated_images, self.rotated_contours), 1):
            plt.subplot(1, len(self.rotated_images), i)
            plt.imshow(image)
            plt.scatter(contour[:, 0], contour[:, 1], c="r", s=2)
            plt.axis("off")
        plt.savefig(self.save_dir.joinpath("02_prealignment.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return

    def finetune_reconstruction(self) -> None:
        """
        Method to finetune the match between adjacent images using lightglue.
        """

        # We use the mid slice as reference point and move all images toward this slice.
        mid_slice = int(np.ceil(len(self.rotated_images)//2))
        self.final_images = [None] * mid_slice + [self.rotated_images[mid_slice]] + [None] * (len(self.rotated_images)-mid_slice-1)
        self.final_masks = [None] * mid_slice + [self.rotated_masks[mid_slice]] + [None] * (len(self.rotated_masks)-mid_slice-1)
        self.final_contours = [None] * mid_slice + [self.rotated_contours[mid_slice]] + [None] * (len(self.rotated_contours)-mid_slice-1)

        self.moving_indices = list(np.arange(0, mid_slice)[::-1]) + list(np.arange(mid_slice+1, len(self.rotated_images)))
        self.ref_indices = list(np.arange(0, mid_slice)[::-1] + 1) + list(np.arange(mid_slice+1, len(self.rotated_images)) - 1)

        # Iteratively perform lightglue matching for adjacent pairs and update the reference points.
        for mov, ref in zip(self.moving_indices, self.ref_indices):

            # Get reference image and extract features 
            ref_image = self.final_images[ref]
            ref_mask = self.final_masks[ref]
            ref_image_tensor = torch.tensor(ref_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
            ref_features = self.extractor.extract(ref_image_tensor)
            
            best_overlap = 0

            # Use 180 degree increments when ellipsoid shape, otherwise 90 degree increments
            ellipse_axis = self.ellipses[mov][1]
            if np.max(ellipse_axis) > 1.25*np.min(ellipse_axis):
                rotations = [0, 180]
            else:
                rotations = [0, 90, 180, 270]

            # Compute keypoints for moving image and moving images 180 deg rotated
            for r in rotations:

                # Get image and apply rotation
                moving_image = self.rotated_images[mov]
                moving_mask = self.rotated_masks[mov]
                moving_contour = self.rotated_contours[mov]

                rotation_matrix = cv2.getRotationMatrix2D(tuple(self.common_center.astype("float")), r, 1)
                moving_image = cv2.warpAffine(moving_image, rotation_matrix, moving_image.shape[:-1][::-1], borderValue=(255, 255, 255))
                moving_mask = cv2.warpAffine(moving_mask, rotation_matrix, moving_mask.shape[::-1], borderValue=(0, 0, 0))
                moving_contour = np.squeeze(cv2.transform(np.expand_dims(moving_contour, axis=0), rotation_matrix))

                # Convert to tensor and extract features
                moving_image_tensor = torch.tensor(moving_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
                moving_features = self.extractor.extract(moving_image_tensor)
                
                # Find matches with features from reference image
                matches01 = self.matcher({'image0': ref_features, 'image1': moving_features})
                ref_features2, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
                matches = matches01['matches']

                # Get keypoints from both images
                points_ref = ref_features2['keypoints'][matches[..., 0]]
                points_moving = moving_features['keypoints'][matches[..., 1]]

                # Save image of corresponding matches
                viz2d.plot_images([ref_image, moving_image])
                viz2d.plot_matches(points_ref, points_moving, color='lime', lw=0.2)
                viz2d.save_plot(self.save_dir.joinpath("keypoints", f"keypoints_{mov}_to_{ref}_rot_{r}.png"))

                # Convert back to numpy
                points_ref = points_ref.cpu().numpy()
                points_moving = points_moving.cpu().numpy()
                ref_image = ref_image_tensor.cpu().numpy().transpose((1, 2, 0))
                moving_image = moving_image_tensor.cpu().numpy().transpose((1, 2, 0))

                # Obtain full affine transformation or solely rotation + scaling
                tform = "partial_affine"
                if tform == "full_affine":
                    rows, cols, _ = moving_image.shape
                    affine_matrix, _ = cv2.estimateAffine2D(points_moving, points_ref)            

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
                moving_mask_warped = cv2.warpAffine(
                        moving_mask, 
                        affine_matrix,  
                        (cols, rows), 
                        borderValue=(0, 0, 0)
                )
                moving_mask_warped = ((moving_mask_warped > 128)*255).astype("uint8")
                moving_contour = np.squeeze(cv2.transform(np.expand_dims(moving_contour, axis=0), affine_matrix))

                # Compute which part of the smallest mask falls within the other mask
                all_mask = [moving_mask_warped, ref_mask]
                min_idx = np.argmin([np.sum(i) for i in all_mask])
                overlap = np.sum(all_mask[min_idx] & all_mask[1-min_idx]) / np.sum(all_mask[min_idx])

                if overlap > best_overlap:
                    best_overlap = overlap

                    # Save final image and contours
                    self.final_images[mov] = moving_image_warped.astype("uint8")
                    self.final_masks[mov] = moving_mask_warped.astype("uint8")
                    self.final_contours[mov] = moving_contour.astype("int")

                # Get contours for visualization purposes
                cnt_moving, _ = cv2.findContours(moving_mask_warped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cnt_moving = np.squeeze(max(cnt_moving, key=cv2.contourArea))
                cnt_ref, _ = cv2.findContours(ref_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cnt_ref = np.squeeze(max(cnt_ref, key=cv2.contourArea))

                # Save image of result 
                plt.figure(figsize=(20, 10))
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
                plt.savefig(self.save_dir.joinpath("warps", f"warped_{mov}_rot_{r}.png"), dpi=300, bbox_inches="tight")
                plt.close()

        self.final_reconstruction = np.stack(self.final_images, axis=-1)
        self.final_reconstruction_mask = np.stack(self.final_masks, axis=-1)

        # Save final reconstruction
        plt.figure(figsize=(20, 10))
        for i in range(self.final_reconstruction.shape[-1]):
            plt.subplot(1, self.final_reconstruction.shape[-1], i+1)
            plt.imshow(self.final_reconstruction[:, :, :, i])
            plt.axis("off")
        plt.savefig(self.save_dir.joinpath("03_final_reconstruction.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Overview figure of the contours
        plt.figure(figsize=(10, 10))
        plt.imshow(np.zeros((self.final_reconstruction.shape[:2])), cmap="gray")
        for cnt in self.final_contours:
            plt.scatter(cnt[:, 0], cnt[:, 1])
        plt.axis("off")
        plt.legend([i.stem for i in self.image_paths])
        plt.savefig(self.save_dir.joinpath("03_final_reconstruction_contours.png"))
        plt.close()

        return


    def plot_3d_volume(self):
        """
        Method to plot all the levels of the 3D reconstructed volume in a single 3D plot.
        """
    
        # Incorporate 10 empty slices between each slice for better visualization
        block_size = 10
        self.final_reconstruction_3d = np.zeros(
            (self.final_reconstruction.shape[0], 
             self.final_reconstruction.shape[1], 
             block_size*(self.final_reconstruction.shape[3]+1),
             self.final_reconstruction.shape[2])
        ).astype("uint8")
        self.final_reconstruction_3d_mask = np.zeros(
            (self.final_reconstruction_mask.shape[0], 
             self.final_reconstruction_mask.shape[1], 
             block_size*(self.final_reconstruction_mask.shape[2]+1))
        ).astype("uint8")

        for i in range(self.final_reconstruction.shape[3]):
            self.final_reconstruction_3d[:, :, block_size*(i+1), :] = self.final_reconstruction[:, :, :, i]
            self.final_reconstruction_3d_mask[:, :, block_size*(i+1)] = self.final_reconstruction_mask[:, :, i]
        
        self.final_reconstruction_3d[self.final_reconstruction_3d == 255] = 0

        plt.figure()
        plt.subplot(121)
        plt.imshow(self.final_reconstruction_3d[250, :, :, :].transpose(1, 0, 2))
        plt.subplot(122)
        plt.imshow(self.final_reconstruction_3d_mask[250, :, :].transpose(1, 0))
        plt.savefig(self.save_dir.joinpath("3d_reconstruction_slice_test.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # Extract surface mesh from 3D volume
        verts, faces, _, _ = marching_cubes(self.final_reconstruction_3d_mask)

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
            zaxis_title='Z Axis'),
            margin=dict(t=0, b=0, l=0, r=0)
        )

        fig.show()

        return
