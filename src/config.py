class Config:

    def __init__(self):
        
        # The level at which the images are processed for the reconstruction. Lower
        # levels and thus higher resolution are more accurate but also slower.
        self.image_level = 8

        # Difference in number of levels between the original images and the generated tissue masks
        self.image_mask_level_diff = 4

        # Minimum number of images required for a meaningful reconstruction
        self.min_images_for_reconstruction = 3

        # Thresholds for the RANSAC sampling per keypoint detection method. These different
        # detectors produce a different number of keypoints and thresholds are adjusted accordingly.
        self.ransac_thresholds = {
            "lightglue" : 0.05,
            "dalf" : 0.2
        }

        # Fixed ratio for padding the images to a common size.
        self.padding_ratio = 1.2

        # Some specimen sectioning variables. The slice thickness represents the thickness (micron) of one 
        # finalized slice of tissue on a glass slide. The slice distance represents the distance between two
        # adjacent slices in micron. In case one glass slide is prepared per tissue block, this is simply 
        # equal to the thickness of the tissue blocks. These values hold for the prostatectomies from our center
        # but may differ for your own data.
        self.slice_thickness = 4 
        self.slice_distance = 4000 

        return

