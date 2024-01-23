import pyvips
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import torch 
import copy 
import SimpleITK as sitk
import subprocess
import shutil

import sys
sys.path.append("/root/DALF_CVPR_2023")

from pathlib import Path
from skimage.measure import marching_cubes
import plotly.graph_objects as go
from typing import Tuple, List
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interpn, interp1d
import torch.nn.functional as F

from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd
from modules.models.DALF import DALF_extractor as DALF
from modules.tps import RANSAC
from modules.tps import pytorch as tps_pth
from modules.tps import numpy as tps_np

from visualization import *
from utils import *
from config import Config


class Hiprova:

    def __init__(self, data_dir: Path, save_dir: Path, detector: str, tform_tps: bool=False, full_resolution_level: int=-1) -> None:
        
        self.config = Config()

        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_dir = save_dir.joinpath("debug")
        self.debug_dir.mkdir(exist_ok=True, parents=True)

        self.tform_tps = tform_tps
        self.full_resolution = full_resolution_level > 0
        self.full_resolution_level_image = full_resolution_level
        self.full_resolution_level_mask = full_resolution_level - self.config.image_mask_level_diff
        self.detector_name = detector.lower()
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
        self.image_level = self.config.image_level
        self.mask_level = self.image_level-self.config.image_mask_level_diff
        assert self.full_resolution_level_image <= self.image_level, "Full resolution level should be lower than image level."
        self.fullres_scaling = 2 ** (self.image_level - self.full_resolution_level_image)

        # Set device for GPU-based keypoint detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ransac_thres_affine = self.config.ransac_thresholds[self.detector_name]
        self.ransac_thres_tps = self.ransac_thres_affine / 2

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
                image = pyvips.Image.new_from_file(str(im_path), level=self.image_level)
            elif im_path.suffix == ".tif":
                image = pyvips.Image.new_from_file(str(im_path), page=self.image_level)
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

        for c, im_path in enumerate(self.image_paths):
                
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
        self.contours = []

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
           
            # Get contour from mask
            contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = np.squeeze(max(contour, key=cv2.contourArea))

            # Apply mask to image
            image[mask == 0] = 255

            # Save masked image 
            self.contours.append(contour)
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

        # Mask image by adding white to image and then casting to uint8, faster than multiplication?
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
        for contour in self.contours:
            
            # Fit ellipse based on contour 
            ellipse = cv2.fitEllipse(contour)
            center, _, rotation = ellipse

            # Correct rotation for opencv/mpl conventions
            self.rotations.append(rotation)
            self.centerpoints.append(center)
            self.ellipses.append(ellipse)

        # Plot resulting ellipse and contour
        plot_ellipses(
            images=self.images, 
            ellipses=self.ellipses, 
            centerpoints=self.centerpoints, 
            rotations=self.rotations, 
            save_dir=self.local_save_dir
        )

        return

    def prealignment(self) -> None:
        """
        Method to match the center of mass and rotations of adjacent images.
        """

        self.rotated_images = []
        self.rotated_masks = []
        self.rotated_contours = []

        # Find common centerpoint of all ellipses to orient towards
        self.common_center = np.mean([i[0] for i in self.ellipses], axis=0).astype("int")
        c = 1
            
        for image, mask, contour, rotation, center in zip(self.images, self.masks, self.contours, self.rotations, self.centerpoints):

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

        if self.full_resolution:
            self.prealignment_fullres()

        # Plot resulting prealignment
        plot_prealignment(
            images=self.rotated_images, 
            contours=self.rotated_contours, 
            save_dir=self.local_save_dir
        )

        return

    def prealignment_fullres(self) -> None:
        """
        Perform the prealignment on the full resolution images.
        """

        if not hasattr(self, "rotated_images_fullres"):
            self.rotated_images_fullres = []
            self.rotated_masks_fullres = []

        for image_fullres, mask_fullres, rotation, center in zip(self.fullres_images, self.fullres_masks, self.rotations, self.centerpoints):

            # Apply rotation in full resolution
            rotation_matrix_fullres = cv2.getRotationMatrix2D(tuple([i*self.fullres_scaling for i in center]), rotation, 1)

            rotated_image_fullres = image_fullres.affine(
                (rotation_matrix_fullres[0, 0], rotation_matrix_fullres[0, 1], rotation_matrix_fullres[1, 0], rotation_matrix_fullres[1, 1]),
                interpolate = pyvips.Interpolate.new("bicubic"),
                odx = rotation_matrix_fullres[0, 2],
                ody = rotation_matrix_fullres[1, 2],
                oarea = (0, 0, image_fullres.width, image_fullres.height),
                background = 255
            )
            rotated_mask_fullres = mask_fullres.affine(
                (rotation_matrix_fullres[0, 0], rotation_matrix_fullres[0, 1], rotation_matrix_fullres[1, 0], rotation_matrix_fullres[1, 1]),
                interpolate = pyvips.Interpolate.new("nearest"),
                odx = rotation_matrix_fullres[0, 2],
                ody = rotation_matrix_fullres[1, 2],
                oarea = (0, 0, mask_fullres.width, mask_fullres.height),
                background = 0
            )

            # Apply translation in full resolution
            translation = self.common_center - center

            translation_matrix_fullres = np.float32([[1, 0, translation[0]*self.fullres_scaling], [0, 1, translation[1]*self.fullres_scaling]])
            translation_matrix_fullres = np.float32([[1, 0, translation[0]*self.fullres_scaling_mask], [0, 1, translation[1]*self.fullres_scaling_mask]])

            rotated_image_fullres = rotated_image_fullres.affine(
                (translation_matrix_fullres[0, 0], translation_matrix_fullres[0, 1], translation_matrix_fullres[1, 0], translation_matrix_fullres[1, 1]),
                interpolate = pyvips.Interpolate.new("bicubic"),
                odx = translation_matrix_fullres[0, 2],
                ody = translation_matrix_fullres[1, 2],
                oarea = (0, 0, rotated_image_fullres.width, rotated_image_fullres.height),
                background = 255
            )
            rotated_mask_fullres = rotated_mask_fullres.affine(
                (translation_matrix_fullres[0, 0], translation_matrix_fullres[0, 1], translation_matrix_fullres[1, 0], translation_matrix_fullres[1, 1]),
                interpolate = pyvips.Interpolate.new("nearest"),
                odx = translation_matrix_fullres[0, 2],
                ody = translation_matrix_fullres[1, 2],
                oarea = (0, 0, rotated_mask_fullres.width, rotated_mask_fullres.height),
                background = 0
            )

            self.rotated_images_fullres.append(rotated_image_fullres)
            self.rotated_masks_fullres.append(rotated_mask_fullres)

        return

    def get_keypoints(self, tform: str, detector: str) -> None:
        """
        Wrapper function to get the keypoints from the chosen detector.
        """

        if detector == "dalf":
            self.get_dalf_keypoints(tform)
        elif detector == "lightglue":
            self.get_lightglue_keypoints(tform)

        return

    def get_dalf_keypoints(self, tform: str) -> None:
        """
        Wrapper function to get the keypoints from DALF.
        """

        # Initialize DALF detector and matcher
        self.dalf_detector = DALF(dev=self.device)
        self.dalf_matcher = cv2.BFMatcher(crossCheck = True)

        # Get reference image and moving image
        ref_image = self.final_images[self.ref]
        moving_image = self.rotated_images[self.mov]
        moving_mask = self.rotated_masks[self.mov]
        moving_contour = self.rotated_contours[self.mov]

        # Apply flipping for regular images
        rotation_matrix = cv2.getRotationMatrix2D(tuple(self.common_center.astype("float")), self.rot, 1)
        moving_image = cv2.warpAffine(moving_image, rotation_matrix, moving_image.shape[:-1][::-1], borderValue=(255, 255, 255))
        moving_mask = cv2.warpAffine(moving_mask, rotation_matrix, moving_mask.shape[::-1], borderValue=(0, 0, 0))
        moving_contour = np.squeeze(cv2.transform(np.expand_dims(moving_contour, axis=0), rotation_matrix))

        # Extract features
        points_ref, desc_ref = self.dalf_detector.detectAndCompute(ref_image)
        points_moving, desc_moving = self.dalf_detector.detectAndCompute(moving_image)
        
        # Match features
        matcher = cv2.BFMatcher(crossCheck = True)
        matches = matcher.match(desc_ref, desc_moving)

        # Apply matches to keypoints
        points_ref_filt = np.float32([points_ref[m.queryIdx].pt for m in matches])
        points_moving_filt = np.float32([points_moving[m.trainIdx].pt for m in matches])

        # Filter matches with RANSAC
        thres = self.ransac_thres_affine if tform == "affine" else self.ransac_thres_tps
        inliers = RANSAC.nr_RANSAC(points_ref_filt, points_moving_filt, self.device, thr = thres)
        ransac_matches = [matches[i] for i in range(len(matches)) if inliers[i]]

        # Plot resulting keypoints and matches
        savepath = self.local_save_dir.joinpath("keypoints", f"{tform}_keypoints_{self.mov}_to_{self.ref}_rot_{self.rot}.png")
        plot_keypoint_pairs(
            ref_image=ref_image, 
            moving_image=moving_image,
            ref_points=points_ref, 
            moving_points=points_moving, 
            matches=matches,
            ransac_matches=ransac_matches,
            savepath=savepath
        )

        # Convert to numpy and save
        self.points_ref = np.float32([i.pt for i in points_ref]) 
        self.points_ref_filt = points_ref_filt
        self.points_moving = np.float32([i.pt for i in points_moving])
        self.points_moving_filt = points_moving_filt
        self.ref_image = ref_image
        self.ref_mask = self.final_masks[self.ref]
        self.moving_image = moving_image
        self.moving_mask = moving_mask
        self.moving_contour = moving_contour
        self.matches = matches

        if self.full_resolution:
            self.get_keypoints_fullres()

        return

    def get_keypoints_fullres(self) -> None:
        """
        Method to perform the preprocessing for the fullres images for keypoint detection.
        """

        # Get full res image and mask
        moving_image_fullres = self.rotated_images_fullres[self.mov]
        moving_mask_fullres = self.rotated_masks_fullres[self.mov]

        # Apply flipping for fullres images
        if self.rot != 0:
            rotation_matrix_fullres = cv2.getRotationMatrix2D(
                tuple([float(i*self.fullres_scaling) for i in self.common_center]), 
                self.rot, 
                1
            )
            moving_image_fullres = moving_image_fullres.affine(
                (rotation_matrix_fullres[0, 0], rotation_matrix_fullres[0, 1], rotation_matrix_fullres[1, 0], rotation_matrix_fullres[1, 1]),
                interpolate = pyvips.Interpolate.new("bicubic"),
                odx = rotation_matrix_fullres[0, 2],
                ody = rotation_matrix_fullres[1, 2],
                oarea = (0, 0, moving_image_fullres.width, moving_image_fullres.height),
                background = 255
            )
            moving_mask_fullres = moving_mask_fullres.affine(
                (rotation_matrix_fullres[0, 0], rotation_matrix_fullres[0, 1], rotation_matrix_fullres[1, 0], rotation_matrix_fullres[1, 1]),
                interpolate = pyvips.Interpolate.new("nearest"),
                odx = rotation_matrix_fullres[0, 2],
                ody = rotation_matrix_fullres[1, 2],
                oarea = (0, 0, moving_mask_fullres.width, moving_mask_fullres.height),
                background = 0
            )

        self.moving_image_fullres = moving_image_fullres
        self.moving_mask_fullres = moving_mask_fullres

        return

    def get_lightglue_keypoints(self, tform: str) -> None:
        """
        Wrapper function to get the keypoints from lightglue.
        """

        # Initialize lightglue detector and matcher
        self.lightglue_detector = SuperPoint(max_num_keypoints=2048).eval().cuda()  
        self.lightglue_matcher = LightGlue(features='superpoint').eval().cuda() 

        # Get reference image and extract features 
        ref_image = self.final_images[self.ref]
        ref_image_tensor = torch.tensor(ref_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
        ref_features = self.lightglue_detector.extract(ref_image_tensor)

        # Get image and apply rotation
        moving_image = self.rotated_images[self.mov]
        moving_mask = self.rotated_masks[self.mov]
        moving_contour = self.rotated_contours[self.mov]

        # Apply flipping for regular images
        rotation_matrix = cv2.getRotationMatrix2D(tuple(self.common_center.astype("float")), self.rot, 1)
        moving_image = cv2.warpAffine(moving_image, rotation_matrix, moving_image.shape[:-1][::-1], borderValue=(255, 255, 255))
        moving_mask = cv2.warpAffine(moving_mask, rotation_matrix, moving_mask.shape[::-1], borderValue=(0, 0, 0))
        moving_contour = np.squeeze(cv2.transform(np.expand_dims(moving_contour, axis=0), rotation_matrix))

        # Convert to tensor and extract features
        moving_image_tensor = torch.tensor(moving_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
        moving_features = self.lightglue_detector.extract(moving_image_tensor)
        
        # Find matches with features from reference image
        matches01 = self.lightglue_matcher({'image0': ref_features, 'image1': moving_features})
        ref_features2, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
        matches = matches01['matches']
        scores = [np.round(float(i), 5) for i in matches01['scores'].detach().cpu().numpy()]

        # Get keypoints from both images
        points_ref_filt = np.float32([i.astype("int") for i in ref_features2['keypoints'][matches[..., 0]].cpu().numpy()])
        points_moving_filt = np.float32([i.astype("int") for i in moving_features['keypoints'][matches[..., 1]].cpu().numpy()])

        # Filter matches with RANSAC
        thres = self.ransac_thres_affine if tform == "affine" else self.ransac_thres_tps
        inliers = RANSAC.nr_RANSAC(points_ref_filt, points_moving_filt, self.device, thr = thres)
        ransac_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
        ransac_scores = [scores[i] for i in range(len(matches)) if inliers[i]]

        # Modify some variables for proper plotting
        savepath = self.local_save_dir.joinpath("keypoints", f"{tform}_keypoints_{self.mov}_to_{self.ref}_rot_{self.rot}.png")
        ref_points_plot = ref_features2['keypoints'].cpu().numpy()
        ref_points_plot = [cv2.KeyPoint(x, y, 1) for x, y in ref_points_plot]
        moving_points_plot = moving_features['keypoints'].cpu().numpy()
        moving_points_plot = [cv2.KeyPoint(x, y, 1) for x, y in moving_points_plot]

        plot_matches = [cv2.DMatch(int(a.cpu()), int(b.cpu()), s) for (a, b), s in zip(matches, scores)]
        plot_ransac_matches = [cv2.DMatch(int(a.cpu()), int(b.cpu()), s) for (a, b), s in zip(ransac_matches, ransac_scores)]

        plot_keypoint_pairs(
            ref_image=ref_image, 
            moving_image=moving_image,
            ref_points=ref_points_plot, 
            moving_points=moving_points_plot, 
            matches=plot_matches,
            ransac_matches=plot_ransac_matches,
            savepath=savepath
        )

        # Convert back to numpy and save
        self.points_ref = ref_features2['keypoints'].cpu().numpy()
        self.points_ref_filt = points_ref_filt
        self.points_moving = moving_features['keypoints'].cpu().numpy()
        self.points_moving_filt = points_moving_filt
        self.ref_image = self.final_images[self.ref]
        self.ref_mask = self.final_masks[self.ref]
        self.moving_image = (moving_image_tensor.cpu().numpy().transpose((1, 2, 0))*255).astype("uint8")
        self.moving_mask = moving_mask
        self.moving_contour = moving_contour
        self.matches = [cv2.DMatch(int(a), int(b), s) for (a, b), s in zip(matches.cpu(), scores)]

        if self.full_resolution:
            self.get_keypoints_fullres()

        return 

    def apply_transform(self, tform: str = "affine", ransac: bool = True) -> None:
        """
        Wrapper function to apply the transform based on the keypoints found.
        """

        # Apply RANSAC filtering
        if ransac:

            # Apply ransac to further filter plausible matches
            thres = self.ransac_thres_affine if tform == "affine" else self.ransac_thres_tps
            inliers = RANSAC.nr_RANSAC(self.points_ref_filt, self.points_moving_filt, self.device, thr = thres)
            ransac_matches = [self.matches[i] for i in range(len(self.matches)) if inliers[i]]

            self.points_ref_filt = np.float32([self.points_ref[m.queryIdx] for m in ransac_matches])
            self.points_moving_filt = np.float32([self.points_moving[m.trainIdx] for m in ransac_matches])

        # Apply affine or TPS transform
        if tform == "tps":
            self.apply_tps_transform()
        elif tform == "affine":
            self.apply_affine_transform()

        return

    def apply_transform_fullres(self, tform: str = "affine") -> None:
        """
        Wrapper function to apply the transform based on the keypoints found.
        """

        # Apply affine or TPS transform
        if tform == "tps":
            self.apply_tps_transform_fullres()
        elif tform == "affine":
            self.apply_affine_transform_fullres()

        return

    def apply_tps_transform(self) -> None:
        """
        Compute and apply thin plate splines transform.
        """

        # Get image shapes
        h1, w1 = self.ref_image.shape[:2]
        h2, w2 = self.moving_image.shape[:2]
        
        # Normalize coordinates
        c_ref = np.float32(self.points_ref_filt) / np.float32([w1,h1])
        c_moving = np.float32(self.points_moving_filt) / np.float32([w2,h2])

        # Compute theta from coordinates
        moving_image = torch.tensor(self.moving_image).to(self.device).permute(2,0,1)[None, ...].float()
        self.theta = tps_np.tps_theta_from_points(c_ref, c_moving, reduced=True, lambd=0.01)
        self.theta = torch.tensor(self.theta).to(self.device)[None, ...]

        # Create grid to sample from
        self.grid = tps_pth.tps_grid(self.theta, torch.tensor(c_moving, device=self.device), moving_image.shape)

        # Apply transform
        moving_image = F.grid_sample(moving_image, self.grid, align_corners=False)
        self.moving_image_warped = moving_image[0].permute(1,2,0).cpu().numpy().astype(np.uint8)

        # Also apply to mask
        moving_mask = torch.tensor(self.moving_mask).to(self.device)[None, None, ...].float()
        moving_mask = F.grid_sample(moving_mask, self.grid, align_corners=False)
        self.moving_mask_warped = moving_mask[0, 0].cpu().numpy().astype(np.uint8)
        self.moving_mask_warped = ((self.moving_mask_warped > 0)*255).astype("uint8")

        # Compute new contour of mask
        self.moving_contour, _ = cv2.findContours(self.moving_mask_warped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.moving_contour = np.squeeze(max(self.moving_contour, key=cv2.contourArea))

        # Multiply image by mask to get rid of black borders
        self.moving_image_warped[self.moving_mask_warped < np.max(self.moving_mask_warped)] = 255

        return

    def apply_tps_transform_fullres(self) -> None:
        """
        Apply thin plate splines transform to the full resolution.
        """

        # Convert torch grid to pyivps grid
        dx = self.grid.cpu().numpy()[0, :, :, 0]
        dx = ((dx + 1) / 2) * (self.moving_image_fullres.width - 1)

        dy = self.grid.cpu().numpy()[0, :, :, 1] 
        dy = ((dy + 1) / 2) * (self.moving_image_fullres.height - 1)

        # Scale to full resolution
        dx_fullres = pyvips.Image.new_from_array(dx).resize(self.fullres_scaling)
        dy_fullres = pyvips.Image.new_from_array(dy).resize(self.fullres_scaling)
        index_map = dx_fullres.bandjoin([dy_fullres])

        # Apply to image
        self.moving_image_fullres_warped = self.moving_image_fullres.mapim(
            index_map, 
            interpolate=pyvips.Interpolate.new('bicubic'), 
            background=[255, 255, 255]
        )

        # Apply TPS to grid for visualization purposes
        warped_grid = visualize_grid(image_size=self.moving_image.shape, grid=self.grid)

        plt.figure()
        plt.subplot(141)
        plt.imshow(self.moving_image)
        plt.axis("off")
        plt.title("original")
        plt.subplot(142)
        plt.imshow(warped_grid)
        plt.axis("off")
        plt.title("tps grid")
        plt.subplot(143)
        plt.imshow(self.moving_image_warped)
        plt.axis("off")
        plt.title("tps numpy")
        plt.subplot(144)
        plt.imshow(self.moving_image_fullres_warped.numpy())
        plt.axis("off")
        plt.title("tps fullres")
        plt.savefig(self.debug_dir.joinpath("tps_numpy_vs_fullres.png"))
        plt.close()

        return

    def apply_affine_transform(self) -> None:
        """
        Compute a limited affine transform based on rotation and translation.
        """

        # Compute centroids
        centroid_fixed = np.mean(self.points_ref_filt, axis=0)
        centroid_moving = np.mean(self.points_moving_filt, axis=0)

        # Shift the keypoints so that both sets have a centroid at the origin
        points_ref_centered = self.points_ref_filt - centroid_fixed
        points_moving_centered = self.points_moving_filt - centroid_moving

        # Compute the rotation matrix
        H = np.dot(points_moving_centered.T, points_ref_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Create the combined rotation and translation matrix
        affine_matrix = np.zeros((2, 3))
        affine_matrix[:2, :2] = R
        affine_matrix[:2, 2] = centroid_fixed - np.dot(R, centroid_moving)

        # Actually warp the images
        rows, cols, _ = self.moving_image.shape
        self.moving_image_warped = cv2.warpAffine(
            self.moving_image, 
            affine_matrix,  
            (cols, rows), 
            borderValue=(255, 255, 255)
        )
        self.moving_mask_warped = cv2.warpAffine(
            self.moving_mask, 
            affine_matrix,  
            (cols, rows), 
            borderValue=(0, 0, 0)
        )
        self.moving_mask_warped = ((self.moving_mask_warped > 128)*255).astype("uint8")

        # Warp contour 
        self.moving_contour = np.squeeze(cv2.transform(np.expand_dims(self.moving_contour, axis=0), affine_matrix))

        return
    
    def apply_affine_transform_fullres(self) -> None:
        """
        Method to apply the affine transform to the full resolution images.
        """

        # Compute centroids
        centroid_fixed = np.mean(self.points_ref_filt, axis=0)
        centroid_fixed = [i * self.fullres_scaling for i in centroid_fixed]
        centroid_moving = np.mean(self.points_moving_filt, axis=0)
        centroid_moving = [i * self.fullres_scaling for i in centroid_moving]

        # Shift the keypoints so that both sets have a centroid at the origin
        points_ref_filt = np.float32([i * self.fullres_scaling for i in self.points_ref_filt])
        points_moving_filt = np.float32([i * self.fullres_scaling for i in self.points_moving_filt])

        points_ref_centered = points_ref_filt - centroid_fixed
        points_moving_centered = points_moving_filt - centroid_moving

        # Compute the rotation matrix
        H = np.dot(points_moving_centered.T, points_ref_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Create the combined rotation and translation matrix
        affine_matrix = np.zeros((2, 3))
        affine_matrix[:2, :2] = R
        affine_matrix[:2, 2] = centroid_fixed - np.dot(R, centroid_moving)

        # Actually warp the images
        self.moving_image_fullres_warped = self.moving_image_fullres.affine(
            (affine_matrix[0, 0], affine_matrix[0, 1], affine_matrix[1, 0], affine_matrix[1, 1]),
            interpolate = pyvips.Interpolate.new("bicubic"),
            odx = affine_matrix[0, 2],
            ody = affine_matrix[1, 2],
            oarea = (0, 0, self.moving_image_fullres.width, self.moving_image_fullres.height),
            background = 255
        )
        self.moving_mask_fullres_warped = self.moving_mask_fullres.affine(
            (affine_matrix[0, 0], affine_matrix[0, 1], affine_matrix[1, 0], affine_matrix[1, 1]),
            interpolate = pyvips.Interpolate.new("nearest"),
            odx = affine_matrix[0, 2],
            ody = affine_matrix[1, 2],
            oarea = (0, 0, self.moving_mask_fullres.width, self.moving_mask_fullres.height),
            background = 0
        )

        return

    def reconstruction(self, tform: str, detector: str, correct_flip: bool) -> None:
        """
        Method to apply either affine or tps reconstruction.
        """

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

        self.tform_matrices = [None] * len(self.rotated_images)
        self.best_rotations = [None] * len(self.rotated_images)

        # Iteratively perform keypoint matching for adjacent pairs and update the reference points.
        for self.mov, self.ref in zip(self.moving_indices, self.ref_indices):

            self.best_overlap = 0

            # Optional correction for tissue flips
            if correct_flip:
                ellipse_axis = self.ellipses[self.mov][1]
                rotations = np.arange(0, 181, 180) if np.max(ellipse_axis) > 1.25*np.min(ellipse_axis) else np.arange(0, 360, 45)
            else:
                rotations = [0]

            for self.rot in rotations:

                # Get keypoints from either lightglue or dalf
                self.get_keypoints(tform=tform, detector=detector)

                # Apply transform based on keypoints, optionally use RANSAC filtering
                self.apply_transform(tform=tform, ransac=False)
                
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
                    self.best_rotations[self.mov] = self.rot

                    # Perform full resolution reconstruction
                    if self.full_resolution:
                        self.reconstruction_fullres(tform=tform)

                # Plot warped images as sanity check
                save_path = self.local_save_dir.joinpath("warps", f"{tform}_warped_{self.mov}_rot_{self.rot}.png")
                plot_warped_images(
                    self.ref_image, 
                    self.ref_mask,
                    self.moving_image, 
                    self.moving_image_warped, 
                    self.moving_mask_warped,
                    overlap, 
                    save_path
                )

        return

    def reconstruction_fullres(self, tform: str) -> None:
        """
        Method to apply either affine or tps reconstruction on the full resolution images.
        """

        # Preallocate list if not yet created
        if not hasattr(self, "final_images_fullres"):
            mid_slice = int(np.ceil(len(self.rotated_images)//2))
            self.final_images_fullres = [None] * len(self.rotated_images)
            self.final_images_fullres[mid_slice] = self.rotated_images_fullres[mid_slice]

            self.final_masks_fullres = [None] * len(self.rotated_images)
            self.final_masks_fullres[mid_slice] = self.rotated_masks_fullres[mid_slice]

        # Apply transform and save
        self.apply_transform_fullres(tform=tform)
        self.final_images_fullres[self.mov] = self.moving_image_fullres_warped
        self.final_masks_fullres[self.mov] = self.moving_mask_fullres_warped

        return

    def perform_reconstruction(self) -> None:
        """
        Method to finetune the match between adjacent images using keypoint detection and matching.
        By default we use affine reconstruction and optionally TPS for finetuning.
        """

        s = " + tps" if self.tform_tps else ""
        print(f" - finetuning reconstruction [affine{s}]")

        # Always perform affine reconstruction
        self.reconstruction(tform="affine", detector=self.detector_name, correct_flip=True)
        self.final_reconstruction = np.stack(self.final_images, axis=-1)
        plot_final_reconstruction(
            final_reconstruction = self.final_reconstruction, 
            final_contours = self.final_contours, 
            image_paths = self.image_paths, 
            save_dir = self.local_save_dir, 
            tform = "affine"
        )

        # Optionally perform thin-plate-spline finetuning
        if self.tform_tps:

            # Hacky way to repeat same reconstruction method by infusing affine results
            self.rotated_images = copy.copy(self.final_images)
            self.rotated_masks = copy.copy(self.final_masks)
            self.rotated_contours = copy.copy(self.final_contours)
            self.final_images = []
            self.final_masks = []
            self.final_contours = []

            if self.full_resolution:
                self.rotated_images_fullres = copy.copy(self.final_images_fullres)
                self.rotated_masks_fullres = copy.copy(self.final_masks_fullres)
                del self.final_images_fullres, self.final_masks_fullres

            self.reconstruction(tform="tps", detector=self.detector_name, correct_flip=False)
            self.final_reconstruction = np.stack(self.final_images, axis=-1)
            plot_final_reconstruction(
                final_reconstruction = self.final_reconstruction, 
                final_contours = self.final_contours, 
                image_paths = self.image_paths, 
                save_dir = self.local_save_dir, 
                tform = "tps"
            )

        self.final_reconstruction_mask = np.stack(self.final_masks, axis=-1)

        return

    def create_3d_volume(self) -> None:
        """
        Method to create a 3D representation of the stacked slices. 
        Slice thickness is 3 µm and distance between slices is
        3 mm. 
        """

        print(f" - creating 3D volume")

        # Prostate specific variables
        slice_thickness = self.config.slice_thickness
        slice_distance = self.config.slice_distance

        # Get in-plane downsample factor from image level
        self.xy_downsample = self.image_level ** 2

        # Get between plane downsample through tissue characteristics and XY downsample.
        # Block size is the number of empty slices we have to insert between
        # actual size for a representative 3D model.
        self.z_downsample = (slice_distance / slice_thickness) / self.xy_downsample
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

                savepath = self.local_save_dir.joinpath(f"contour_{self.filled_slices[i]+j+1}.png")
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

        print(f" - saving results")

        # Upload local results to external storage 
        subprocess.call(f"cp -r {self.local_save_dir} {self.save_dir.parent}", shell=True)
        shutil.rmtree(self.local_save_dir)

        return
