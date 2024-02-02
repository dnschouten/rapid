import pyvips
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import torch 
import copy 
import subprocess
import shutil
import sys
sys.path.append("/root/DALF_CVPR_2023")

from pathlib import Path
from typing import List

from visualization import *
from utils import *
from config import Config
from transforms import *
from keypoints import *


class Hiprova:

    def __init__(self, data_dir: Path, save_dir: Path, tform_tps: bool=False) -> None:
        
        self.config = Config()

        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_dir = save_dir.joinpath("debug")
        self.debug_dir.mkdir(exist_ok=True, parents=True)

        self.tform_tps = tform_tps
        self.full_resolution = self.config.full_resolution_level > 0
        self.full_resolution_level_image = self.config.full_resolution_level
        self.full_resolution_level_mask = self.config.full_resolution_level - self.config.image_mask_level_diff
        self.detector_name = self.config.detector.lower()
        assert self.detector_name in ["dalf", "lightglue"], "Sorry, only DALF and lightglue detectors are supported."

        self.local_save_dir = Path(f"/tmp/hiprova/{self.save_dir.name}")
        if not self.local_save_dir.is_dir():
            self.local_save_dir.mkdir(parents=True, exist_ok=True)

        # For now only support tif
        self.image_paths = sorted([i for i in self.data_dir.iterdir() if not "mask" in i.name])
        self.image_ids = [i.stem for i in self.image_paths]
        self.mask_paths = sorted([i for i in self.data_dir.iterdir() if "mask" in i.name])

        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks do not match."
        assert len(self.image_paths) > self.config.min_images_for_reconstruction, "Need at least four images to perform a reasonable reconstruction."

        # Create directories
        self.local_save_dir.joinpath("keypoints").mkdir(parents=True, exist_ok=True)
        self.local_save_dir.joinpath("warps").mkdir(parents=True, exist_ok=True)

        # Set level at which to load the image
        self.keypoint_level = self.config.keypoint_level
        self.tps_level = self.config.tps_level
        self.evaluation_level = self.config.evaluation_level
        self.mask_level = self.keypoint_level-self.config.image_mask_level_diff
        assert self.full_resolution_level_image <= self.keypoint_level, "Full resolution level should be lower than image level."
        self.fullres_scaling = 2 ** (self.keypoint_level - self.full_resolution_level_image)
        self.tps_scaling = 2 ** (self.tps_level - self.keypoint_level)

        # Set device for GPU-based keypoint detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ransac_thres_affine = self.config.ransac_thresholds[self.detector_name]
        self.ransac_thres_tps = self.ransac_thres_affine / 2

        self.affine_ransac = self.config.affine_ransac
        self.tps_ransac = self.config.tps_ransac

        return
    
    def load_images(self) -> None:
        """
        Method to load images using pyvips.
        """

        print(f" - loading {len(self.image_paths)} images")
        self.raw_images = []
        
        for im_path in self.image_paths:

            # Load image
            if im_path.suffix == ".mrxs":
                image = pyvips.Image.new_from_file(str(im_path), level=self.keypoint_level)
            elif im_path.suffix == ".tif":
                image = pyvips.Image.new_from_file(str(im_path), page=self.keypoint_level)
            else:
                raise ValueError("Sorry, only .tifs and .mrxs are supported.")

            # Dispose of alpha band if present
            if image.bands == 4:
                image_np = image.flatten().numpy().astype(np.uint8)
            elif image.bands == 3: 
                image_np = image.numpy().astype(np.uint8)

            # Save images
            self.raw_images.append(image_np)

        # Plot initial reconstruction
        plot_initial_reconstruction(
            images=self.raw_images, 
            save_dir=self.local_save_dir
        )

        if self.full_resolution:
            self.load_images_fullres()

        return 

    def load_images_fullres(self) -> None:
        """
        Method to load the full res images.
        """

        self.raw_fullres_images = [] 

        for im_path in self.image_paths:
                
            # Load image
            if im_path.suffix == ".mrxs":
                image_fullres = pyvips.Image.new_from_file(str(im_path), level=self.full_resolution_level_image)
            elif im_path.suffix == ".tif":
                image_fullres = pyvips.Image.new_from_file(str(im_path), page=self.full_resolution_level_image)
            else:
                raise ValueError("Sorry, only .tifs and .mrxs are supported.")

            # Dispose of alpha band if present
            if image_fullres.bands == 4:
                image_fullres = image_fullres.flatten().cast("uchar")

            # Save images
            self.raw_fullres_images.append(image_fullres)

        return

    def load_masks(self) -> None:
        """
        Method to load the masks. These are based on the tissue segmentation algorithm described in:
        Bándi P, Balkenhol M, van Ginneken B, van der Laak J, Litjens G. 2019. Resolution-agnostic tissue segmentation in
        whole-slide histopathology images with convolutional neural networks. PeerJ 7:e8242 DOI 10.7717/peerj.8242.
        """

        self.raw_masks = []
        
        for c, mask_path in enumerate(self.mask_paths):

            # Load mask and convert to numpy array
            mask = pyvips.Image.new_from_file(str(mask_path), page=self.mask_level)
            mask_np = mask.numpy()

            # Ensure size match between mask and image
            im_shape = self.raw_images[c].shape[:2]
            if im_shape[0] != mask_np.shape[0]:
                mask_np = cv2.resize(mask_np, (im_shape[1], im_shape[0]), interpolation=cv2.INTER_NEAREST)

            # Save numpy mask
            mask_np = ((mask_np > 0)*255).astype(np.uint8)
            self.raw_masks.append(mask_np)

        if self.full_resolution:
            self.load_masks_fullres()

        return

    def load_masks_fullres(self) -> None:
        """
        Method to load the full res masks.
        """

        self.raw_fullres_masks = [] 

        for c, mask_path in enumerate(self.mask_paths):
                
            # Load mask and convert to numpy array
            mask_fullres = pyvips.Image.new_from_file(str(mask_path), page=self.full_resolution_level_mask)
            
            self.fullres_scaling_mask = mask_fullres.width / self.raw_fullres_images[c].width
            if self.fullres_scaling_mask != 1:
                mask_fullres = mask_fullres.resize(1/self.fullres_scaling_mask)

            # Save numpy mask
            mask_fullres = ((mask_fullres > 0)*255).cast("uchar")
            self.raw_fullres_masks.append(mask_fullres)

        return

    def apply_masks(self) -> None:
        """
        Method to mask the images based on the convex hull of the contour. This allows
        for some information to be retained outside of the prostate.
        """

        self.images = []
        self.masks = []

        # Get common size
        factor = self.config.padding_ratio
        max_h = int(np.max([i.shape[0] for i in self.raw_images]) * factor)
        max_w = int(np.max([i.shape[1] for i in self.raw_images]) * factor)
        c=1

        for c, (image, mask) in enumerate(zip(self.raw_images, self.raw_masks)):
            
            # Pad numpy image to common size
            h1, h2 = int(np.ceil((max_h - image.shape[0]) / 2)), int(np.floor((max_h - image.shape[0]) / 2))
            w1, w2 = int(np.ceil((max_w - image.shape[1]) / 2)), int(np.floor((max_w - image.shape[1]) / 2))
            image = np.pad(image, ((h1, h2), (w1, w2), (0, 0)), mode="constant", constant_values=255)
            mask = np.pad(mask, ((h1, h2), (w1, w2)), mode="constant", constant_values=0)
            mask = ((mask > 0)*255).astype("uint8")

            # Apply mask to image
            image[mask == 0] = 255

            # Save masked image 
            self.images.append(image)
            self.masks.append(mask)

            if self.full_resolution:
                self.apply_masks_fullres(c, h1, w1, max_h, max_w)

        return

    def apply_masks_fullres(self, c: int, h1: int, w1: int, max_h: int, max_w: int) -> None:
        """
        Method to apply the full resolution masks.
        """

        if not hasattr(self, "fullres_images"):
            self.fullres_images = []
            self.fullres_masks = []

        fullres_image = self.raw_fullres_images[c]
        fullres_mask = self.raw_fullres_masks[c]

        # Pad pyvips images to common size
        fullres_image = fullres_image.embed(
            w1 * self.fullres_scaling, 
            h1 * self.fullres_scaling, 
            max_w * self.fullres_scaling, 
            max_h * self.fullres_scaling, 
            extend="white"
        )
        fullres_mask = fullres_mask.embed(
            w1 * self.fullres_scaling, 
            h1 * self.fullres_scaling, 
            max_w * self.fullres_scaling, 
            max_h * self.fullres_scaling, 
            extend="black"
        )
        inverse_fullres_mask = 255 - fullres_mask

        # Mask image by adding white to image and then casting to uint8
        masked_image_fullres = (fullres_image + inverse_fullres_mask).cast("uchar")
        self.fullres_images.append(masked_image_fullres)
        self.fullres_masks.append(fullres_mask)

        return

    def find_rotations(self) -> None:
        """
        Method to get the rotation of the prostate based on an
        ellipsoid approximating the fit of the prostate.
        """

        print(f" - performing prealignment")
        self.rotations = []
        self.centerpoints = []
        self.ellipses = []

        # Find ellipse for all images
        for mask in self.masks:
            
            # Get contour from mask
            contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = np.squeeze(max(contour, key=cv2.contourArea))

            # Fit ellipse based on contour 
            ellipse = cv2.fitEllipse(contour)
            center, _, rotation = ellipse

            # Correct rotation for opencv/mpl conventions
            self.rotations.append(rotation)
            self.centerpoints.append(center)
            self.ellipses.append(ellipse)

        # Plot resulting ellipse
        plot_ellipses(
            images=self.images, 
            ellipses=self.ellipses, 
            centerpoints=self.centerpoints, 
            rotations=self.rotations, 
            save_dir=self.local_save_dir
        )

        return

    def prealignment(self, images: List[np.ndarray], masks: List[np.ndarray], images_fullres: List[pyvips.Image], masks_fullres: List[pyvips.Image]) -> tuple([List, List, List, List]):
        """
        Step to align the images based on an ellipse fitted through the prostate.
        """

        # As part of the prealignment we need to find the rotation for each fragment.
        self.find_rotations()

        final_images = []
        final_masks = []
        final_images_fullres = []
        final_masks_fullres = []

        # Find common centerpoint of all ellipses to orient towards
        self.common_center = np.mean([i[0] for i in self.ellipses], axis=0).astype("int")

        for image, mask, image_fullres, mask_fullres, rotation, center in zip(images, masks, images_fullres, masks_fullres, self.rotations, self.centerpoints):

            # Adjust rotation 
            rotation_matrix = cv2.getRotationMatrix2D(tuple(center), rotation, 1)
            rotated_image, rotated_mask = apply_affine_transform(
                image = image,
                tform = rotation_matrix,
                mask = mask,
            )

            # Adjust translation 
            translation = self.common_center - center
            translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
            rotated_image, rotated_mask = apply_affine_transform(
                image = rotated_image,
                tform = translation_matrix,
                mask = rotated_mask,
            )

            final_images.append(rotated_image)
            final_masks.append(rotated_mask)

            if self.full_resolution:
                # Apply rotation in full resolution
                rotated_image_fullres, rotated_mask_fullres = apply_affine_transform_fullres(
                    image = image_fullres,
                    mask = mask_fullres,
                    rotation = rotation,
                    translation = [0, 0],
                    center = center,
                    scaling = self.fullres_scaling
                )

                # Apply translation in full resolution
                translation = self.common_center - center
                rotated_image_fullres, rotated_mask_fullres = apply_affine_transform_fullres(
                    image = rotated_image_fullres,
                    mask = rotated_mask_fullres,
                    rotation = 0,
                    translation = translation,
                    center = list(self.common_center),
                    scaling = self.fullres_scaling
                )

                final_images_fullres.append(rotated_image_fullres)
                final_masks_fullres.append(rotated_mask_fullres)

        # Plot resulting prealignment
        plot_prealignment(
            images=final_images, 
            save_dir=self.local_save_dir
        )

        return final_images, final_masks, final_images_fullres, final_masks_fullres

    def affine_reconstruction(self, images: List[np.ndarray], masks: List[np.ndarray], images_fullres: List[pyvips.Image], masks_fullres: List[pyvips.Image]) -> tuple([List, List, List, List]):
        """
        Method to perform the affine reconstruction between adjacent slides. This
        step consists of:
            1) Computing keypoints and matches between a pair of images.
            2) Finding the optimal orientation of the moving image.
            3) Using the keypoint to find an affine transform.
            4) Extrapolating this transform to the full res images.

        The affine transform is a limited transform that only includes 
        rotation and translation. 
        """

        print(f" - performing affine reconstruction")

        # We use the mid slice as reference point and move all images toward this slice.
        mid_slice = int(np.ceil(len(images)//2))
        final_images = [None] * len(images)
        final_images[mid_slice] = images[mid_slice]
        final_masks = [None] * len(images)
        final_masks[mid_slice] = masks[mid_slice]
        final_images_fullres = [None] * len(images)
        final_images_fullres[mid_slice] = images_fullres[mid_slice]
        final_masks_fullres = [None] * len(images)
        final_masks_fullres[mid_slice] = masks_fullres[mid_slice]

        moving_indices = list(np.arange(0, mid_slice)[::-1]) + list(np.arange(mid_slice+1, len(images)))
        moving_indices = list(map(int, moving_indices))
        ref_indices = list(np.arange(0, mid_slice)[::-1] + 1) + list(np.arange(mid_slice+1, len(images)) - 1)
        ref_indices = list(map(int, ref_indices))

        for mov, ref in zip(moving_indices, ref_indices):

            best_num_matches = 0

            ellipse_axis = self.ellipses[mov][1]
            rotations = np.arange(0, 181, 180) if np.max(ellipse_axis) > 1.25*np.min(ellipse_axis) else np.arange(0, 360, 45)

            for rot in rotations:

                # Compute flipped version of image
                rotation_matrix = cv2.getRotationMatrix2D(tuple(self.common_center.astype("float")), rot, 1)
                moving_image, moving_mask = images[mov], masks[mov]
                moving_image, moving_mask = apply_affine_transform(
                    image = moving_image,
                    tform = rotation_matrix,
                    mask = moving_mask,
                )
                ref_image = final_images[ref]

                # Extract keypoints
                ref_points, moving_points = get_keypoints(
                    detector = self.detector_name, 
                    ref_image = ref_image, 
                    moving_image = moving_image
                )
                plot_keypoint_pairs(
                    ref_image = ref_image,
                    moving_image = moving_image,
                    ref_points = ref_points,
                    moving_points = moving_points,
                    tform = "affine",
                    savepath = self.local_save_dir.joinpath("keypoints", f"keypoints_affine_{mov}_to_{ref}_rot_{rot}.png")
                )

                # Apply transforms
                affine_matrix = estimate_affine_transform(
                    moving_points = moving_points, 
                    ref_points = ref_points, 
                    image = moving_image, 
                    ransac = self.affine_ransac, 
                    ransac_thres = self.ransac_thres_affine
                )
                moving_image_warped, moving_mask_warped = apply_affine_transform(
                    image = moving_image,
                    tform = affine_matrix.params[:-1, :],
                    mask = moving_mask,
                )

                # Plot resulting warp
                plot_warped_images(
                    ref_image = ref_image,
                    ref_mask = final_masks[ref],
                    moving_image = moving_image,
                    moving_image_warped = moving_image_warped,
                    moving_mask_warped = moving_mask_warped,
                    savepath = self.local_save_dir.joinpath("warps", f"warps_affine_{mov}_to_{ref}_rot_{rot}.png")
                )

                if len(ref_points) > best_num_matches:
                    best_num_matches = len(ref_points)

                    # Save final image
                    final_images[mov] = moving_image_warped.astype("uint8")
                    final_masks[mov] = moving_mask_warped.astype("uint8")

                    # Perform full resolution reconstruction
                    if self.full_resolution:
                        moving_image_fullres_warped, moving_mask_fullres_warped = apply_affine_transform_fullres(
                            image = images_fullres[mov],
                            mask = masks_fullres[mov],
                            rotation = rot,
                            translation = [0, 0],
                            center = self.common_center,
                            scaling = self.fullres_scaling
                        )
                        moving_image_fullres_warped, moving_mask_fullres_warped = apply_affine_transform_fullres(
                            image = moving_image_fullres_warped,
                            mask = moving_mask_fullres_warped,
                            rotation = -math.degrees(affine_matrix.rotation),
                            translation = affine_matrix.translation,
                            center = [0, 0],
                            scaling = self.fullres_scaling
                        )
                        final_images_fullres[mov] = moving_image_fullres_warped
                        final_masks_fullres[mov] = moving_mask_fullres_warped

        return final_images, final_masks, final_images_fullres, final_masks_fullres

    def tps_reconstruction(self, images: List[np.ndarray], masks: List[np.ndarray], images_fullres: List[pyvips.Image], masks_fullres: List[pyvips.Image]) -> tuple([List, List, List, List]):
        """
        Method to perform a deformable reconstruction between adjacent slides. This
        step consists of:
            1) Computing keypoints and matches between a pair of images.
            2) Using the keypoint to find a thin plate splines transform.
            3) Extrapolating this transform to the full res images.

        In principle we can use any deformable registration here as long as the
        deformation can be represented as a grid to warp the image.
        """

        print(f" - performing deformable reconstruction")

        # We use the mid slice as reference point and move all images toward this slice.
        mid_slice = int(np.ceil(len(images)//2))
        final_images = [None] * len(images)
        final_images[mid_slice] = images[mid_slice]
        final_masks = [None] * len(images)
        final_masks[mid_slice] = masks[mid_slice]
        final_images_fullres = [None] * len(images)
        final_images_fullres[mid_slice] = images_fullres[mid_slice]
        final_masks_fullres = [None] * len(images)
        final_masks_fullres[mid_slice] = masks_fullres[mid_slice]

        moving_indices = list(np.arange(0, mid_slice)[::-1]) + list(np.arange(mid_slice+1, len(images)))
        moving_indices = list(map(int, moving_indices))
        ref_indices = list(np.arange(0, mid_slice)[::-1] + 1) + list(np.arange(mid_slice+1, len(images)) - 1)
        ref_indices = list(map(int, ref_indices))

        for mov, ref in zip(moving_indices, ref_indices):

            # Compute flipped version of image
            moving_image, moving_mask = images[mov], masks[mov]
            ref_image = images[ref]

            # Extract keypoints
            ref_points, moving_points = get_keypoints(
                detector = self.detector_name, 
                ref_image = ref_image, 
                moving_image = moving_image
            )
            plot_keypoint_pairs(
                ref_image = ref_image,
                moving_image = moving_image,
                ref_points = ref_points,
                moving_points = moving_points,
                tform = "tps",
                savepath = self.local_save_dir.joinpath("keypoints", f"keypoints_tps_{mov}_to_{ref}.png")
            )

            # Apply transforms
            index_map, grid = estimate_tps_transform(
                moving_image = moving_image,
                ref_image = ref_image,
                moving_points = moving_points, 
                ref_points = ref_points, 
                ransac = self.tps_ransac, 
                ransac_thres_tps = self.ransac_thres_tps,
                tps_level = self.tps_level,
                keypoint_level = self.keypoint_level,
                device = self.device
            )
            moving_image_warped, moving_mask_warped = apply_tps_transform(
                moving_image = moving_image,
                moving_mask = moving_mask,
                index_map = index_map    
            )
            plot_warped_tps_images(
                ref_image = ref_image,
                moving_image = moving_image,
                moving_image_warped = moving_image_warped,
                grid = grid,
                savepath = self.debug_dir.joinpath(f"warps_tps_{mov}_to_{ref}.png")
            )

            # Save final image
            final_images[mov] = moving_image_warped.astype("uint8")
            final_masks[mov] = moving_mask_warped.astype("uint8")

            # Perform full resolution reconstruction
            if self.full_resolution:
                moving_image_fullres_warped, moving_mask_fullres_warped = apply_tps_transform_fullres(
                    image = images_fullres[mov],
                    mask = masks_fullres[mov],
                    grid = grid,
                    scaling = self.tps_scaling
                )
                final_images_fullres[mov] = moving_image_fullres_warped
                final_masks_fullres[mov] = moving_mask_fullres_warped

        return final_images, final_masks, final_images_fullres, final_masks_fullres


    def perform_reconstruction(self) -> None:
        """
        Method to perform all steps of the reconstruction process.
        """

        # Pre-alignment as initial step
        images, masks, fullres_images, fullres_masks = self.prealignment(
            images = self.images,
            masks = self.masks,
            images_fullres = self.fullres_images,
            masks_fullres = self.fullres_masks
        )

        # Affine registration to improve prealignment
        images, masks, fullres_images, fullres_masks = self.affine_reconstruction(
            images = images,
            masks = masks,
            images_fullres = fullres_images,
            masks_fullres = fullres_masks,
        )
        plot_final_reconstruction(
            final_images = images, 
            save_dir = self.local_save_dir, 
            tform = "affine"
        )

        # Deformable registration as final step
        if self.tform_tps:
            images, masks, fullres_images, fullres_masks = self.tps_reconstruction(
                images = images,
                masks = masks,
                images_fullres = fullres_images,
                masks_fullres = fullres_masks,
            )
            plot_final_reconstruction(
                final_images = images, 
                save_dir = self.local_save_dir, 
                tform = "tps"
            )

        return

    def create_3d_volume(self) -> None:
        """
        Method to create a 3D representation of the stacked slices. We leverage sectioning
        variables such as slice thickness, slice distance and x-y-z downsampling levels
        to create an anatomical true to size 3D volume.
        """

        print(f" - creating 3D volume")

        # Prostate specific variables
        slice_thickness = self.config.slice_thickness
        slice_distance = self.config.slice_distance

        # Downsample images again in order to create 3D volume
        partial_xy_downsample = 2 ** (self.evaluation_level - self.keypoint_level)
        total_xy_downsample = 2 ** self.evaluation_level
        new_size = tuple(int(i / partial_xy_downsample) for i in self.final_images[0].shape[:2][::-1])
        
        self.final_images = [cv2.resize(i, new_size, interpolation=cv2.INTER_AREA) for i in self.final_images]
        self.final_masks = [cv2.resize(i, new_size, interpolation=cv2.INTER_NEAREST) for i in self.final_masks]
        self.final_contours = [(c / partial_xy_downsample).astype(np.int32) for c in self.final_contours]

        # Block size is the number of empty slices we have to insert between
        # actual slices for a true to size 3D model.
        z_downsample = slice_distance / total_xy_downsample
        self.block_size = int(np.round(z_downsample * slice_thickness)-1)

        # Pre-allocate 3D volumes 
        self.final_reconstruction_3d = np.zeros(
            (self.final_images[0].shape[0], 
             self.final_images[0].shape[1], 
             self.block_size*(len(self.final_images)+1),
             self.final_images[0].shape[2]),
             dtype="uint8"
        )
        self.final_reconstruction_3d_mask = np.zeros(
            (self.final_masks[0].shape[0],  
             self.final_masks[0].shape[1],  
             self.block_size*(len(self.final_masks)+1)),                  
             dtype="uint8"
        )

        # Populate actual slices
        for c, (im, mask) in enumerate(zip(self.final_images, self.final_masks)):
            self.final_reconstruction_3d[:, :, self.block_size*(c+1), :] = im
            self.final_reconstruction_3d_mask[:, :, self.block_size*(c+1)] = mask
            
        self.final_reconstruction_3d[self.final_reconstruction_3d == 255] = 0

        return
    
    def interpolate_3d_volume(self) -> None:
        """
        Method to interpolate the 2D slices to a binary 3D volume.
        """

        self.final_reconstruction_volume = copy.copy(self.final_reconstruction_3d_mask)
        self.filled_slices = [self.block_size*(i+1) for i in range(len(self.final_images))]

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

                savepath = self.local_save_dir.joinpath(f"contour_{self.filled_slices[i]+j+1}.png")
                # plot_interpolated_contour(slice_a, contour_a, mask, contour, slice_b, contour_b, savepath)

                self.final_reconstruction_volume[:, :, self.filled_slices[i]+j+1] = mask

        return

    def evaluate_reconstruction(self):
        """
        Method to compute the metrics to evaluate the reconstruction quality.
        """

        print(" - evaluating reconstruction")

        tre = compute_tre_keypoints()

        # Compute sphericity of the reconstructed volume
        # self.sphericity = compute_sphericity(self.final_reconstruction_volume)

        # Compute dice score of all adjacent masks
        # reconstruction_dice = compute_reconstruction_dice(masks = self.final_masks)

        return

    def save_results(self):
        """
        Copy all created figures from Docker to external storage.
        """

        print(f" - saving results")

        # Upload local results to external storage 
        subprocess.call(f"cp -r {self.local_save_dir} {self.save_dir.parent}", shell=True)
        shutil.rmtree(self.local_save_dir)

        return
    
    def run(self):
        """
        Method to run the full pipeline.
        """

        self.load_images()
        self.load_masks()
        self.apply_masks()
        self.perform_reconstruction()
        # self.create_3d_volume()
        # self.interpolate_3d_volume()
        # self.plot_3d_volume()
        # self.evaluate_reconstruction()
        self.save_results()

        return
