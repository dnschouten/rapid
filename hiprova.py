import pyvips
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import multiresolutionimageinterface as mir
from scipy import ndimage
from pathlib import Path


class Hiprova:


    def __init__(self, data_dir: Path, save_dir: Path) -> None:
        
        # Set data+save directories
        self.data_dir = data_dir
        self.save_dir = save_dir.joinpath(data_dir.name)

        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # For now only support mrxs
        self.slides = self.data_dir.glob("*.mrxs")
        self.slides = sorted(list(self.slides))

        # Set level at which to load the image
        self.level = 9

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
        thres = 250

        for image_hsv in self.images_hsv:

            # Threshold the v-channel
            mask = image_hsv[:, :, 2] < thres
            mask = ndimage.binary_fill_holes(mask).astype("uint8")

            # Only keep the largest component in the mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = ((labels == largest_label)*255).astype("uint8")
            
            self.masks.append(mask)

        return

    def get_centerpoints(self) -> None:
        """
        Method to get the centerpoint of the masks.
        """

        self.centerpoints = []
        self.contours = []

        for mask in self.masks:

            # Get contour
            contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = np.squeeze(max(contour, key=cv2.contourArea))

            # Compute moments of contour to get centerpoint
            moments = cv2.moments(contour)
            centerpoint = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

            self.centerpoints.append(centerpoint)
            self.contours.append(contour)

        return
    
    def get_rotations(self) -> None:
        """
        Method to get the rotation of the prostate based on the 
        eigenvectors of the contour of the mask.
        """

        self.rotations = []
        self.eigenvectors = []

        for contour in self.contours:
            
            # Get eigenvectors and angle
            _, eigenvector = cv2.PCACompute(np.squeeze(contour).reshape(-1, 2).astype("uint16"), mean=np.empty((0)))
            angle = np.degrees(np.arctan2(eigenvector[0, 1], eigenvector[0, 0]))

            self.rotations.append(np.round(angle, 2))
            self.eigenvectors.append(eigenvector)  

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

        # Convert all rotated images to grayscale and stack them into a 3D volume
        self.test = np.stack([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in self.rotated_images], axis=-1)

        return

    def finetune_reconstruction(self) -> None:
        """
        Function to finetune the match between adjacent images using the 
        contour of adjacent images.
        """

        mid_slice = int(np.ceil(len(self.rotated_images)//2))
        self.finetuned_images = [None] * mid_slice + [self.rotated_images[mid_slice]] + [None] * (len(self.rotated_images)-mid_slice-1)
        self.finetuned_contours = [None] * mid_slice + [self.rotated_contours[mid_slice]] + [None] * (len(self.rotated_contours)-mid_slice-1)


        # Loop over all slices above the mid slice and finetune the reconstruction
        for i in np.arange(0, mid_slice)[::-1]:

            # Get reference image and contour
            ref_contour = self.rotated_contours[i+1]
            ref_contour_pc = open3d.geometry.PointCloud()
            ref_contour_pc.points = open3d.utility.Vector3dVector(np.squeeze(ref_contour))

            # Get the current contour and image
            src_contour = self.rotated_contours[i]
            src_image = self.rotated_images[i]
            src_contour_pc = open3d.geometry.PointCloud()
            src_contour_pc.points = open3d.utility.Vector3dVector(np.squeeze(src_contour))

            # Perform ICP registration to go from src to ref
            icp_result = o3d.pipelines.registration.registration_icp(
                src_contour_pc, 
                ref_contour_pc, 
                threshold=0.01, 
                max_correspondence_distance=0.01,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=400000, max_validation=500)
            )

            # Apply transformation to src image and contour
            transformation_matrix = icp_result.transformation
            src_image = cv2.warpAffine(src_image, transformation_matrix[:2], src_image.shape[:-1][::-1], borderValue=(255, 255, 255))
            src_contour = cv2.transform(np.expand_dims(src_contour, axis=0), transformation_matrix)

            # Save finetuned image and contour in list
            self.finetuned_images[i] = src_image
            self.finetuned_contours[i] = src_contour
        
        # Loop over all slices below the mid slice and finetune the reconstruction
        for i in np.arange(mid_slice+1, len(self.rotated_images)):

            # Get reference image and contour
            ref_contour = self.rotated_contours[i-1]
            ref_contour_pc = open3d.geometry.PointCloud()
            ref_contour_pc.points = open3d.utility.Vector3dVector(np.squeeze(ref_contour))

            # Get the current contour and image
            src_contour = self.rotated_contours[i]
            src_image = self.rotated_images[i]
            src_contour_pc = open3d.geometry.PointCloud()
            src_contour_pc.points = open3d.utility.Vector3dVector(np.squeeze(src_contour))

            # Perform ICP registration to go from src to ref
            icp_result = o3d.pipelines.registration.registration_icp(
                src_contour_pc, 
                ref_contour_pc, 
                threshold=0.01, 
                max_correspondence_distance=0.01,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=400000, max_validation=500)
            )

            # Apply transformation to src image and contour
            transformation_matrix = icp_result.transformation
            src_image = cv2.warpAffine(src_image, transformation_matrix[:2], src_image.shape[:-1][::-1], borderValue=(255, 255, 255))
            src_contour = cv2.transform(np.expand_dims(src_contour, axis=0), transformation_matrix)

            # Save finetuned image and contour in list
            self.finetuned_images[i] = src_image
            self.finetuned_contours[i] = src_contour

        return
    