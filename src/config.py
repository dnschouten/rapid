class Config:

    def __init__(self):
        
        # Apply translation and rotation to make it more difficult.
        self.scramble = True

        # Name of the keypoint detection method. Must be in ["dalf", "sift", "superpoint", "loftr", "aspanformer", "roma", "dedode", "omniglue"].
        self.detector = "omniglue"

        # The level at which the images are processed for the different parts of the reconstruction. Lower
        # levels and thus higher resolution are more accurate but also cause memory problems.
        # > keypoint level: level at which the keypoints are detected and affine transform is computed
        # > deformable level: level at which the deformable transform is computed
        # > evaluation level: level at which the reconstruction evaluation is performed
        self.keypoint_level = 8
        self.deformable_level = 8
        self.evaluation_level = 9

        # Whether the affine/deformable transform is computed using RANSAC.
        self.affine_ransac = True
        self.deformable_ransac = False

        # Difference in number of levels between the original images and the generated tissue masks
        self.image_mask_level_diff = 4

        # The level at which the full resolution images are reconstructed. For full 
        # resolution choose 0, for skipping this component choose -1. 
        self.full_resolution_level = 7
        self.normalize_dice = True

        # Minimum number of images required for a meaningful reconstruction
        self.min_images_for_reconstruction = 3

        # Thresholds for the RANSAC sampling per keypoint detection method. These different
        # detectors produce a different number of keypoints and thresholds are adjusted accordingly.
        self.ransac_thresholds = {
            "superpoint" : 0.05,
            "sift": 0.05,
            "dalf" : 0.2,
            "disk": 0.05,
            "loftr": 0.05,
            "aspanformer": 0.05,
            "roma": 0.05,
            "dedode": 0.05,
            "omniglue": 0.05,
        }

        # Some specimen sectioning variables. The slice thickness represents the thickness (micron) of one 
        # finalized slice of tissue on a glass slide. The slice distance represents the distance between two
        # adjacent slices in micron. In case one glass slide is prepared per tissue block, this is simply 
        # equal to the thickness of the tissue blocks. These values hold for the prostatectomies from our center
        # but may differ for your own data.
        self.slice_thickness = 4 
        self.slice_distance = 4000 

        return

