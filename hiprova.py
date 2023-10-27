import pyvips
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import torch 
import copy 
import SimpleITK as sitk

import sys
sys.path.append("/root/DALF_CVPR_2023")

from scipy import ndimage
from pathlib import Path
from scipy.spatial import procrustes, distance
from skimage.measure import marching_cubes
import plotly.graph_objects as go
from sklearn.metrics import normalized_mutual_info_score
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, List
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interpn, interp1d

from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd
from modules.models.DALF import DALF_extractor as DALF

from visualization import *
from utils import *


class Hiprova:

    def __init__(self, data_dir: Path, mask_dir: Path, save_dir: Path, detector: str) -> None:
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.mask_dir = mask_dir
        self.detector_name = detector.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # For now only support tif
        self.image_paths = self.data_dir.glob("*.tif")
        self.image_paths = sorted(list(self.image_paths))
        self.mask_paths = [self.mask_dir.joinpath(i.name) for i in self.image_paths]

        # Create directories
        self.save_dir.joinpath("keypoints").mkdir(parents=True, exist_ok=True)
        self.save_dir.joinpath("warps").mkdir(parents=True, exist_ok=True)

        # Initialize keypoint detector and matcher
        if self.detector_name == "dalf":
            self.dalf_detector = DALF(dev=self.device)
            self.dalf_matcher = cv2.BFMatcher(crossCheck = True)
        elif self.detector_name == "lightglue":
            self.lightglue_detector = SuperPoint(max_num_keypoints=2048).eval().cuda()  
            self.lightglue_matcher = LightGlue(features='superpoint').eval().cuda()
        else:
            raise ValueError("Sorry, only DALF and lightglue detectors are supported.")

        # Set level at which to load the image
        self.image_level = 9
        self.mask_level = self.image_level-3

        return
    
    def load_images(self) -> None:
        """
        Method to load images using pyvips.
        """

        self.images = []
        
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

            # Save images
            self.images.append(image)

        # Plot initial reconstruction
        plot_initial_reconstruction(
            images=self.images, 
            save_dir=self.save_dir
        )

        return 

    def load_masks(self) -> None:
        """
        Method to load the masks. These are based on the tissue segmentation algorithm described in:
        Bándi P, Balkenhol M, van Ginneken B, van der Laak J, Litjens G. 2019. Resolution-agnostic tissue segmentation in
        whole-slide histopathology images with convolutional neural networks. PeerJ 7:e8242 DOI 10.7717/peerj.8242.
        """

        self.masks = []
        
        for mask_path in self.mask_paths:

            # Hacky workaround to prevent pyvips invalid tile byte count error
            mask = pyvips.Image.new_from_file(str(mask_path), page=self.mask_level)
            try:
                mask = mask.numpy()
            except:
                mask = pyvips.Image.new_from_file(str(mask_path), page=self.mask_level+1)
                mask = mask.resize(2).numpy()

            mask_needs_processing = True
            
            if mask_needs_processing:
                # Fill some holes 
                mask = ndimage.binary_fill_holes(mask).astype("uint8")

                # Smooth the mask a bit
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # Only keep the largest component in the mask
                _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = ((labels == largest_label)*255).astype("uint8")

            else:
                mask = ((mask / np.max(mask)) * 255).astype("uint8")
                
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
            
            mask_type = "regular"
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
        plot_ellipses(
            images=self.masked_images, 
            ellipses=self.ellipses, 
            centerpoints=self.centerpoints, 
            rotations=self.rotations, 
            save_dir=self.save_dir
        )

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
        plot_prealignment(
            images=self.rotated_images, 
            contours=self.rotated_contours, 
            save_dir=self.save_dir
        )

        return

    def get_dalf_keypoints(self) -> None:
        """
        Wrapper function to get the keypoints from DALF.
        """

        # Get reference image and extract features 
        ref_image = self.final_images[self.ref]
        ref_image_tensor = torch.tensor(ref_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()

        return

    def get_lightglue_keypoints(self) -> None:
        """
        Wrapper function to get the keypoints from lightglue.
        """

        # Get reference image and extract features 
        ref_image = self.final_images[self.ref]
        ref_image_tensor = torch.tensor(ref_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
        ref_features = self.extractor.extract(ref_image_tensor)

        # Get image and apply rotation
        moving_image = self.rotated_images[self.mov]
        moving_mask = self.rotated_masks[self.mov]
        moving_contour = self.rotated_contours[self.mov]

        rotation_matrix = cv2.getRotationMatrix2D(tuple(self.common_center.astype("float")), self.rot, 1)
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
        savepath = self.save_dir.joinpath("keypoints", f"keypoints_{self.mov}_to_{self.ref}_rot_{self.rot}.png")
        plot_keypoint_pairs(
            ref_image=ref_image, 
            moving_image=moving_image, 
            ref_points=points_ref, 
            moving_points=points_moving, 
            savepath=savepath
        )

        # Convert back to numpy
        self.points_ref = points_ref.cpu().numpy()
        self.points_moving = points_moving.cpu().numpy()
        self.ref_image = self.final_images[self.ref]
        self.ref_mask = self.final_masks[self.ref]
        self.moving_image = moving_image_tensor.cpu().numpy().transpose((1, 2, 0))
        self.moving_mask = copy.copy(moving_mask)
        self.moving_contour = copy.copy(moving_contour)

        return 
   
    def warp_affine(self) -> None:
        """
        Convenience function to warp images and contours using
        a limited affine transformation (only rotation+translation).
        """

        # Compute centroids
        centroid_fixed = np.mean(self.points_ref, axis=0)
        centroid_moving = np.mean(self.points_moving, axis=0)

        # Shift the keypoints so that both sets have a centroid at the origin
        points_ref_centered = self.points_ref - centroid_fixed
        points_moving_centered = self.points_moving - centroid_moving

        # Compute the rotation matrix
        H = np.dot(points_moving_centered.T, points_ref_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Create the combined rotation and translation matrix
        self.affine_matrix = np.zeros((2, 3))
        self.affine_matrix[:2, :2] = R
        self.affine_matrix[:2, 2] = centroid_fixed - np.dot(R, centroid_moving)

        # Actually warp the images
        rows, cols, _ = self.moving_image.shape
        self.moving_image_warped = cv2.warpAffine(
            (self.moving_image * 255).astype("uint8"), 
            self.affine_matrix,  
            (cols, rows), 
            borderValue=(255, 255, 255)
        )
        self.moving_mask_warped = cv2.warpAffine(
            self.moving_mask, 
            self.affine_matrix,  
            (cols, rows), 
            borderValue=(0, 0, 0)
        )
        self.moving_mask_warped = ((self.moving_mask_warped > 128)*255).astype("uint8")

        # Warp contour
        self.moving_contour = np.squeeze(cv2.transform(np.expand_dims(self.moving_contour, axis=0), self.affine_matrix))

        return


    def warp_tps(self) -> None:
        """
        Convenience function to warp images and contours using
        a thin plate spline transformation.
        """

        # Compute thin plate spline transformation
        self.tps = cv2.createThinPlateSplineShapeTransformer()

        # Add grid corners for well-defined grid
        grid_corners = np.array([
            [0, 0],
            [0, self.moving_image.shape[1]], 
            [0, self.moving_image.shape[0]], 
            [self.moving_image.shape[0], self.moving_image.shape[1]]
        ], dtype=np.float32)
        points_moving_tps = np.vstack([self.points_moving, grid_corners])
        points_ref_tps = np.vstack([self.points_ref, grid_corners])

        # Estimate transformation
        matches = [cv2.DMatch(i, i, 0) for i in range(len(points_moving_tps))] 
        self.tps.setRegularizationParameter(0.0001)
        self.tps.estimateTransformation(
            points_moving_tps.reshape(1, -1, 2), 
            points_ref_tps.reshape(1, -1, 2), 
            matches
        )

        # Warp image and mask
        rows, cols, _ = self.moving_image.shape
        self.moving_image_warped = self.tps.warpImage(self.moving_image, (cols, rows))
        self.moving_mask_warped = self.tps.warpImage(self.moving_mask, (cols, rows))

        return

    def finetune_reconstruction(self, detector: str) -> None:
        """
        Method to finetune the match between adjacent images using lightglue.
        """

        assert detector.lower() in ["dalf", "lightglue"], "only dalf and lightglue are supported"

        # We use the mid slice as reference point and move all images toward this slice.
        mid_slice = int(np.ceil(len(self.rotated_images)//2))
        self.final_images = [None] * len(self.rotated_images)
        self.final_images[mid_slice] = self.rotated_images[mid_slice]
        self.final_masks = [None] * len(self.rotated_images)
        self.final_masks[mid_slice] = self.rotated_masks[mid_slice]
        self.final_contours = [None] * len(self.rotated_images)
        self.final_contours[mid_slice] = self.rotated_contours[mid_slice]

        self.moving_indices = list(np.arange(0, mid_slice)[::-1]) + list(np.arange(mid_slice+1, len(self.rotated_images)))
        self.moving_indices = list(map(int, self.moving_indices))
        self.ref_indices = list(np.arange(0, mid_slice)[::-1] + 1) + list(np.arange(mid_slice+1, len(self.rotated_images)) - 1)
        self.ref_indices = list(map(int, self.ref_indices))

        # Iteratively perform lightglue matching for adjacent pairs and update the reference points.
        for self.mov, self.ref in zip(self.moving_indices, self.ref_indices):


            
            self.best_overlap = 0

            # Use 180 degree increments when ellipsoid shape, otherwise 45 degree increments
            ellipse_axis = self.ellipses[self.mov][1]
            rotations = np.arange(0, 181, 180) if np.max(ellipse_axis) > 1.25*np.min(ellipse_axis) else np.arange(0, 360, 45)

            for self.rot in rotations:

                # Get keypoints from lightglue
                self.get_lightglue_keypoints()
                self.warp_affine()
                # self.warp_tps()

                # Compute which part of the smallest mask falls within the other mask
                all_mask = [self.moving_mask_warped, self.ref_mask]
                min_idx = np.argmin([np.sum(i) for i in all_mask])
                overlap = np.sum(all_mask[min_idx] & all_mask[1-min_idx]) / np.sum(all_mask[min_idx])

                if overlap > self.best_overlap:
                    self.best_overlap = overlap

                    # Save final image and contours
                    self.final_images[self.mov] = self.moving_image_warped.astype("uint8")
                    self.final_masks[self.mov] = self.moving_mask_warped.astype("uint8")
                    self.final_contours[self.mov] = self.moving_contour.astype("int")

                # Plot warped images as sanity check
                save_path = self.save_dir.joinpath("warps", f"warped_{self.mov}_rot_{self.rot}.png")
                plot_warped_images(
                    self.ref_image, 
                    self.ref_mask,
                    self.moving_image, 
                    self.moving_image_warped, 
                    self.moving_mask_warped,
                    overlap, 
                    save_path
                )

                # save_path = self.save_dir.joinpath("warps", f"tps_grid_{self.mov}_rot_{self.rot}.png")
                # plot_tps_grid(self.points_moving, self.points_ref, self.tps, save_path)

                
        self.final_reconstruction = np.stack(self.final_images, axis=-1)
        self.final_reconstruction_mask = np.stack(self.final_masks, axis=-1)

        # Plot final reconstruction
        plot_final_reconstruction(self.final_reconstruction, self.final_contours, self.image_paths, self.save_dir)

        return

    def create_3d_volume(self) -> None:
        """
        Method to create a 3D representation of the stacked slices. 
        Slice thickness is 3 µm and distance between slices is
        3 mm. 
        """

        # Prostate specific variables
        SLICE_THICKNESS = 3
        SLICE_DISTANCE = 3000

        # Get in-plane downsample factor from image level
        self.xy_downsample = self.image_level ** 2

        # Get between plane downsample through tissue characteristics and XY downsample.
        # Block size is the number of empty slices we have to insert between
        # actual size for a representative 3D model.
        self.z_downsample = (SLICE_DISTANCE / SLICE_THICKNESS) / self.xy_downsample
        self.block_size = int(np.round(self.z_downsample)-1)

        # Pre-allocate 3D volumes 
        self.final_reconstruction_3d = np.zeros(
            (self.final_reconstruction.shape[0], 
             self.final_reconstruction.shape[1], 
             self.block_size*(self.final_reconstruction.shape[3]+1),
             self.final_reconstruction.shape[2])
        ).astype("uint8")
        self.final_reconstruction_3d_mask = np.zeros(
            (self.final_reconstruction_mask.shape[0], 
             self.final_reconstruction_mask.shape[1], 
             self.block_size*(self.final_reconstruction_mask.shape[2]+1))
        ).astype("uint8")

        # Populate actual slices
        for i in range(self.final_reconstruction.shape[3]):
            self.final_reconstruction_3d[:, :, self.block_size*(i+1), :] = self.final_reconstruction[:, :, :, i]
            self.final_reconstruction_3d_mask[:, :, self.block_size*(i+1)] = self.final_reconstruction_mask[:, :, i]
            
        self.final_reconstruction_3d[self.final_reconstruction_3d == 255] = 0

        return
    

    def interpolate_3d_volume(self) -> None:
        """
        Method to interpolate the 2D slices to a binary 3D volume.
        """

        self.final_reconstruction_volume = copy.copy(self.final_reconstruction_3d_mask)
        self.filled_slices = [self.block_size*(i+1) for i in range(self.final_reconstruction.shape[3])]

        # Loop over slices for interpolation
        for i in range(len(self.filled_slices)-1):

            # Get two adjacent slices
            slice_a = self.final_reconstruction_3d_mask[:, :, self.filled_slices[i]]
            slice_b = self.final_reconstruction_3d_mask[:, :, self.filled_slices[i+1]]

            # Get contours, simplify and resample
            num_points = 360
            contour_a = self.final_contours[i]
            contour_a = simplify_contour(contour_a)
            contour_a = resample_contour_radial(contour_a, num_points)

            contour_b = self.final_contours[i+1]
            contour_b = simplify_contour(contour_b)
            contour_b = resample_contour_radial(contour_b, num_points)

            for j in range(self.block_size-1):

                # Compute weighted average of contour a and b
                fraction = j / (self.block_size-1)
                contour = (1-fraction) * contour_a + fraction * contour_b

                # Fill contour to make a mask
                mask = np.zeros_like(slice_a)
                cv2.drawContours(mask, [contour.astype("int")], -1, (255),thickness=cv2.FILLED)

                savepath = self.save_dir.joinpath(f"contour_{self.filled_slices[i]+j+1}.png")
                if False:
                    plot_interpolated_contour(slice_a, contour_a, mask, contour, slice_b, contour_b, savepath)

                self.final_reconstruction_volume[:, :, self.filled_slices[i]+j+1] = mask

        return

       
    def plot_3d_volume(self):
        """
        Method to plot all the levels of the 3D reconstructed volume in a single 3D plot.
        """
    
        # Extract surface mesh from 3D volume
        verts, faces, _, _ = marching_cubes(self.final_reconstruction_volume)

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
    
    def save_results(self):
        """
        Copy all created figures from Docker to external storage.
        """


        return
