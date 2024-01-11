# Function to perform ICP registration with open3d
src_contour_pc = open3d.geometry.PointCloud()
src_contour_pc.points = open3d.utility.Vector3dVector(src_contour)

# Perform ICP registration to go from src to ref
icp_result = open3d.pipelines.registration.registration_icp(
    src_contour_pc, 
    ref_contour_pc, 
    threshold=0.01, 
    max_correspondence_distance=0.01,
    estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
    ransac=open3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, max_validation=500)
)

def initial_alignment(self) -> None:
        """
        Meythod to estimate an initial alignment using procrustes.

        DEPRECATED

        """

        # Loop over all images and get the contour
        self.aligned_images = [self.images[0]]
        self.aligned_contours = [self.contours[0]]  

        for i in range(len(self.images)-1):
                
            # Get ref contour and convert to 3D
            ref_contour = self.aligned_contours[i]
            ref_contour = np.concatenate([ref_contour, np.ones((ref_contour.shape[0], 1))], axis=-1)
            ref_image = self.aligned_images[i]

            # Get src contour and convert to 3D
            src_contour = self.contours[i+1]
            src_contour = np.concatenate([src_contour, np.ones((src_contour.shape[0], 1))], axis=-1)
            src_image = self.images[i+1]
            src_centerpoint = self.centerpoints[i+1]

            # Interpolate the smallest contour to have same number of points
            # as the other contour
            if src_contour.shape[0] < ref_contour.shape[0]:
                sample_idx = np.linspace(0, ref_contour.shape[0]-1, src_contour.shape[0]).astype("int")
                ref_contour = ref_contour[sample_idx, :]
            elif src_contour.shape[0] > ref_contour.shape[0]:
                sample_idx = np.linspace(0, src_contour.shape[0]-1, ref_contour.shape[0]).astype("int")
                src_contour = src_contour[sample_idx, :]
                
            # Warp using procrustes algorithm 
            m, test, _ = trimesh.registration.procrustes(
                src_contour, 
                ref_contour, 
                reflection=False, 
                translation=True, 
                scale=True
            )

            plt.figure()
            plt.scatter(ref_contour[:, 0], ref_contour[:, 1], c="r")
            plt.scatter(src_contour[:, 0], src_contour[:, 1], c="b")
            plt.scatter(test[:, 0], test[:, 1], c="g")
            plt.legend(["ref", "src", "test"])
            plt.savefig(self.save_dir.joinpath(f"procrustes_{i}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # Get translation and rotation
            translation = m[:2, 3]
            sx = np.linalg.norm(m[:3, 0])
            sy = np.linalg.norm(m[:3, 1])
            sz = np.linalg.norm(m[:3, 2])
            m_rot = m[:3, :3] / np.array([sx, sy, sz])
            rot = math.degrees(np.arctan2(m_rot[1, 0], m_rot[0, 0]))

            # Create transformation matrix from translation and rotation
            m_tform = cv2.getRotationMatrix2D(tuple(src_centerpoint), rot, 1)
            m_tform[:2, 2] += translation

            # Apply transformation to image and contour
            aligned_image = cv2.warpAffine(src_image, m_tform[:2], src_image.shape[:-1][::-1], borderValue=(255, 255, 255))
            aligned_contour = cv2.transform(np.expand_dims(src_contour[:, :2], axis=0), m_tform)

            self.aligned_images.append(aligned_image)
            self.aligned_contours.append(aligned_contour)

            plt.figure()
            plt.subplot(131)
            plt.imshow(ref_image)
            plt.scatter(ref_contour[:, 0], ref_contour[:, 1], c="r")
            plt.subplot(132)
            plt.imshow(src_image)
            plt.scatter(src_contour[:, 0], src_contour[:, 1], c="r")
            plt.subplot(133)
            plt.imshow(aligned_image)
            plt.scatter(aligned_contour[:, 0], aligned_contour[:, 1], c="r")
            plt.savefig(self.save_dir.joinpath(f"initial_alignment_{i}.png"), dpi=300, bbox_inches="tight")

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

def get_contours(self) -> None:
    """
    Method to get the centerpoint of the masks.

    Deprecated in favour of getting the centerpoint from the fitEllipse method.
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
        contour_hull = np.vstack([contour_hull, contour_hull[0, :]])

        self.centerpoints.append(centerpoint)
        self.contours.append(contour)
        self.contours_hull.append(contour_hull)

    return

def find_rotations(self) -> None:
    """
    Method to get the rotation of the prostate based on the 
    eigenvectors of the contour of the mask.

    DEPRECATED IN FAVOUR OF USING THE FIT ELLIPSE METHOD.
    """

    self.rotations = []

    for i, contour in enumerate(self.contours_hull):
        
        # Get eigenvectors and angle
        _, eigenvector = cv2.PCACompute(np.squeeze(contour).reshape(-1, 2).astype("uint16"), mean=np.empty((0)))
        angle = np.degrees(np.arctan2(eigenvector[1, 1], eigenvector[1, 0]))

        self.rotations.append(np.round(angle, 2))
        self.eigenvectors.append(eigenvector)  

    # Plot the contour and eigenvectors
    plt.figure(figsize=(20, 10))
    for c, (image, contour, eigenvector, cp) in enumerate(zip(self.masked_images, self.contours_hull, self.eigenvectors, self.centerpoints), 1):
        plt.subplot(1, len(self.images), c)
        plt.imshow(image)
        plt.plot(contour[:, 0], contour[:, 1], c="r")
        plt.quiver(*np.array(cp), *eigenvector[0, :], color="g", scale=10)
        plt.quiver(*np.array(cp), *eigenvector[1, :], color="b", scale=10)
        plt.axis("off")
    plt.savefig(self.save_dir.joinpath(f"eigenvectors.png"), dpi=300, bbox_inches="tight")
    plt.close()


def verify_rotations_new(self) -> None:
    """
    Method to verify that all rotations from the preaalign method are correct.
    """

    # Verify the rotations of the contours by checking the angle between adjacent contours
    for i in range(len(self.rotated_images) - 1):

        # Get current and next image and contour
        current_image = self.rotated_images[i]
        next_image = self.rotated_images[i+1]

        # Get contours
        current_contour = np.squeeze(self.rotated_contours[i])
        next_contour = np.squeeze(self.rotated_contours[i+1])
        
        # Convert to grayscale
        # current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        # next_image = cv2.cvtColor(next_image, cv2.COLOR_RGB2GRAY)

        """
        current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)[:, :, 1]
        next_image = cv2.cvtColor(next_image, cv2.COLOR_RGB2HSV)[:, :, 1]

        # Normalize from [0, 255]
        current_image = ((current_image - np.min(current_image)) / (np.max(current_image) - np.min(current_image)) * 255).astype("uint8")
        next_image = ((next_image - np.min(next_image)) / (np.max(next_image) - np.min(next_image)) * 255).astype("uint8")

        # Get next image but then rotated by 180 degrees
        center = np.mean(self.rotated_contours[i+1], axis=0)
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center), 180, 1)
        next_image_rot = cv2.warpAffine(
            next_image, 
            rotation_matrix, 
            next_image.shape[::-1], 
            borderValue=(255, 255, 255)
        ).astype("uint8")

        # Get next contour but then rotated by 180 degrees
        next_contour_rot = np.squeeze(cv2.transform(
            np.expand_dims(np.squeeze(self.rotated_contours[i+1]), axis=0), 
            rotation_matrix
        ))

        # Compute mutual information
        mi_regular = normalized_mutual_info_score(current_image.flatten(), next_image.flatten())
        mi_rot = normalized_mutual_info_score(current_image.flatten(), next_image_rot.flatten())

        # Compute average contour distance
        dist_next = distance.cdist(next_contour, current_contour)
        avg_dist_next = np.mean(np.min(dist_next, axis=-1))

        dist_next_rot = distance.cdist(next_contour_rot, current_contour)
        avg_dist_next_rot = np.mean(np.min(dist_next_rot, axis=-1))

        # Show current image, next image and next image rot
        plt.figure(figsize=(20, 10))
        plt.subplot(141)
        plt.imshow(current_image, cmap="gray")
        plt.scatter(current_contour[:, 0], current_contour[:, 1], c="r", s=2)
        plt.axis("off")
        plt.title("current image")
        
        plt.subplot(142)
        plt.imshow(next_image, cmap="gray")
        plt.scatter(next_contour[:, 0], next_contour[:, 1], c="b", s=2)
        plt.axis("off")
        plt.title(f"next image (MI={mi_regular:.2f})")
        
        plt.subplot(143)
        plt.imshow(next_image_rot, cmap="gray")
        plt.scatter(next_contour_rot[:, 0], next_contour_rot[:, 1], c="g", s=2)
        plt.axis("off")
        plt.title(f"next image rot (MI={mi_rot:.2f})")
        
        plt.subplot(144)
        pad = 350
        background_image = np.ones((current_image.shape[0], current_image.shape[1]+pad)).astype("uint8")*255
        background_image[-1, -1] = 0
        plt.imshow(background_image, cmap=r"gray")
        plt.plot(current_contour[:, 0], current_contour[:, 1], c="r", lw=2)
        plt.plot(current_contour[:, 0]+pad, current_contour[:, 1], c="r", lw=2)
        plt.plot(next_contour[:, 0], next_contour[:, 1], c="b", lw=1)
        plt.plot(next_contour_rot[:, 0]+pad, next_contour_rot[:, 1], c="g", lw=1)
        plt.axis("off")
        plt.title(f"avg dist current-next: {avg_dist_next:.2f} \n"
                    f"avg dist current-next_rot: {avg_dist_next_rot:.2f}")

        plt.savefig(self.save_dir.joinpath(f"MI_{i}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        """

        plt.figure(figsize=(20, 10))
        channels = np.arange(0, 3)
        channel_names = ["H", "S", "V"]

        for channel, name in zip(channels, channel_names):
            current_image_c = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)[:, :, channel]
            next_image_c = cv2.cvtColor(next_image, cv2.COLOR_RGB2HSV)[:, :, channel]

            # Normalize from [0, 255]
            current_image_c = ((current_image_c - np.min(current_image_c)) / (np.max(current_image_c) - np.min(current_image_c)) * 255).astype("uint8")
            next_image_c = ((next_image_c - np.min(next_image_c)) / (np.max(next_image_c) - np.min(next_image_c)) * 255).astype("uint8")

            # Get next image but then rotated by 180 degrees
            center = np.mean(self.rotated_contours[i+1], axis=0)
            rotation_matrix = cv2.getRotationMatrix2D(tuple(center), 180, 1)
            next_image_rot = cv2.warpAffine(
                next_image_c, 
                rotation_matrix, 
                next_image_c.shape[::-1], 
                borderValue=(255, 255, 255)
            ).astype("uint8")

            # Get next contour but then rotated by 180 degrees
            next_contour_rot = np.squeeze(cv2.transform(
                np.expand_dims(np.squeeze(self.rotated_contours[i+1]), axis=0), 
                rotation_matrix
            ))

            # Compute mutual information
            mi_regular = normalized_mutual_info_score(current_image_c.flatten(), next_image_c.flatten())
            mi_rot = normalized_mutual_info_score(current_image_c.flatten(), next_image_rot.flatten())

            # Compute average contour distance
            dist_next = distance.cdist(next_contour, current_contour)
            avg_dist_next = np.mean(np.min(dist_next, axis=-1))

            dist_next_rot = distance.cdist(next_contour_rot, current_contour)
            avg_dist_next_rot = np.mean(np.min(dist_next_rot, axis=-1))

            # Show current image, next image and next image rot
            plt.subplot(3, 4, channel*4+1)
            plt.imshow(current_image_c, cmap="gray")
            plt.scatter(current_contour[:, 0], current_contour[:, 1], c="r", s=2)
            plt.axis("off")
            plt.title(f"current image {name}")
            
            plt.subplot(3, 4, channel*4+2)
            plt.imshow(next_image_c, cmap="gray")
            plt.scatter(next_contour[:, 0], next_contour[:, 1], c="b", s=2)
            plt.axis("off")
            plt.title(f"next image {name} (MI={mi_regular:.2f})")
            
            plt.subplot(3, 4, channel*4+3)
            plt.imshow(next_image_rot, cmap="gray")
            plt.scatter(next_contour_rot[:, 0], next_contour_rot[:, 1], c="g", s=2)
            plt.axis("off")
            plt.title(f"next image {name} rot (MI={mi_rot:.2f})")
            
            plt.subplot(3, 4, channel*4+4)
            pad = 350
            background_image = np.ones((current_image_c.shape[0], current_image_c.shape[1]+pad)).astype("uint8")*255
            background_image[-1, -1] = 0
            plt.imshow(background_image, cmap=r"gray")
            plt.plot(current_contour[:, 0], current_contour[:, 1], c="r", lw=2)
            plt.plot(current_contour[:, 0]+pad, current_contour[:, 1], c="r", lw=2)
            plt.plot(next_contour[:, 0], next_contour[:, 1], c="b", lw=1)
            plt.plot(next_contour_rot[:, 0]+pad, next_contour_rot[:, 1], c="g", lw=1)
            plt.axis("off")
            plt.title(f"avg dist current-next: {avg_dist_next:.2f} \n"
                    f"avg dist current-next_rot: {avg_dist_next_rot:.2f}")

        plt.savefig(self.save_dir.joinpath(f"MI_{i}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    return

def verify_rotations(self) -> None:
    """
    Method to verify that all rotations are correct.
    """

    # Verify the rotations of the contours by checking the angle between adjacent contours
    for i in range(len(self.rotated_contours) - 1):

        # Get current and next contour
        current_contour = np.squeeze(self.rotated_contours[i])
        next_contour = np.squeeze(self.rotated_contours[i+1])
        next_image = self.rotated_images[i+1]

        mode = "simple"
        if mode == "procrustes":
            # Ensure that the contours are of the same size
            if current_contour.shape[0] > next_contour.shape[0]:
                sample_idx = np.linspace(0, current_contour.shape[0]-1, next_contour.shape[0]).astype("int")
                current_contour = current_contour[sample_idx, :]
            elif current_contour.shape[0] < next_contour.shape[0]:
                sample_idx = np.linspace(0, next_contour.shape[0]-1, current_contour.shape[0]).astype("int")
                next_contour = next_contour[sample_idx, :]

            # Compute procrustes to find best fit
            mtx1, mtx2, _ = procrustes(current_contour, next_contour)

            # Extract rotation
            rotation_matrix = np.dot(mtx1.T, mtx2)
            angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            angle_deg = np.degrees(angle_rad)

            
            # If the angle is closer to 180 than 0, we need to rotate the next contour by 180 degrees
            if np.abs(angle_deg) > 90:

                # Get rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(tuple(next_contour.mean(axis=0).astype("int16")), 180, 1)

                # Apply rotation to contour and image
                rotated_contour = cv2.transform(
                    np.expand_dims(np.squeeze(self.rotated_contours[i+1]), axis=0), 
                    rotation_matrix
                )
                rotated_image = cv2.warpAffine(next_image, rotation_matrix, next_image.shape[:-1][::-1], borderValue=(255, 255, 255))

                # Save rotated contour and image
                self.rotated_contours[i+1] = rotated_contour
                self.rotated_images[i+1] = rotated_image
        
        elif mode == "simple":
        
            needs_rotation = False

            # Compute the shortest distance from each point in the next contour to the current contour
            distances = distance.cdist(next_contour, current_contour)
            avg_distances = np.mean(np.min(distances, axis=-1))

            # Rotate next contour by 180 along its centerpoint
            rotation_matrix = cv2.getRotationMatrix2D(tuple(next_contour.mean(axis=0).astype("int16")), 180, 1)
            next_contour_rot = cv2.transform(
                np.expand_dims(np.squeeze(next_contour), axis=0), 
                rotation_matrix
            )
            next_contour_rot = np.squeeze(next_contour_rot)

            # 
            distances_rot = distance.cdist(next_contour_rot, current_contour)
            avg_distances_rot = np.mean(np.min(distances_rot, axis=-1))

            if avg_distances_rot < avg_distances:
                
                next_image_rot = cv2.warpAffine(next_image, rotation_matrix, next_image.shape[:-1][::-1], borderValue=(255, 255, 255))
                self.rotated_images[i+1] = next_image_rot
                self.rotated_contours[i+1] = next_contour_rot
                needs_rotation = True

                # Save rotated contour and image
                self.rotated_contours[i+1] = next_contour_rot
                self.rotated_images[i+1] = next_image_rot
                
            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.scatter(next_contour[:, 0], next_contour[:, 1], c="r")
            plt.scatter(current_contour[:, 0], current_contour[:, 1], c="b")
            plt.title("Before rot verification")
            plt.axis("off")
            plt.subplot(122)
            plt.scatter(next_contour_rot[:, 0], next_contour_rot[:, 1], c="r")
            plt.scatter(current_contour[:, 0], current_contour[:, 1], c="b")
            plt.title(f"After rot verification, 180 rot {needs_rotation}")
            plt.axis("off")
            plt.savefig(self.save_dir.joinpath("procrustes", f"procrustes_{i}.png"), dpi=300, bbox_inches="tight")
            plt.close()


    # Get non zero areas of contour and crop images based on these indices for efficiency
    # pad = 0.05
    # self.xmin = np.min([np.min(contour[:, 1]) for contour in self.rotated_contours])
    # self.xmax = np.max([np.max(contour[:, 1]) for contour in self.rotated_contours])
    # self.xmin = np.max([0, self.xmin-(pad*(self.xmax-self.xmin))])
    # self.xmax = np.min([self.rotated_images[0].shape[1], self.xmax+(pad*(self.xmax-self.xmin))])
    # self.ymin = np.min([np.min(contour[:, 0]) for contour in self.rotated_contours])
    # self.ymax = np.max([np.max(contour[:, 0]) for contour in self.rotated_contours])
    # self.ymin = np.max([0, self.ymin-(pad*(self.ymax-self.ymin))])
    # self.ymax = np.min([self.rotated_images[0].shape[0], self.ymax+(pad*(self.ymax-self.ymin))])

    # self.rotated_images = [image[int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax)] for image in self.rotated_images]    
    # self.rotated_contours = [contour - np.array([self.xmin, self.ymin]) for contour in self.rotated_contours]

    # Stack all rotated images into a 3D volume
    self.initial_reconstruction = np.stack(self.rotated_images, axis=-1)

    # Plot initial reconstruction
    plt.figure(figsize=(20, 10))
    for i in range(self.initial_reconstruction.shape[-1]):
        plt.subplot(1, self.initial_reconstruction.shape[-1], i+1)
        plt.imshow(self.initial_reconstruction[:, :, :, i])
        plt.axis("off")
    plt.savefig(self.save_dir.joinpath("03_initial_reconstruction_v2.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return

def finetune_reconstruction_icp(self) -> None:
    """
    Method to finetune the match between adjacent images using the 
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

    def finetune_reconstruction(self) -> None:
        """
        Method to finetune the match between adjacent images using lightglue.
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
            # matches02 = self.matcher({'image0': ref_features, 'image1': moving_features2})
            ref_features, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
            # ref_features2, moving_features2, matches02 = [rbd(x) for x in [ref_features, moving_features2, matches02]]
            matches = matches01['matches']
            # matches2 = matches02['matches']
            points_ref = ref_features['keypoints'][matches[..., 0]]
            # points_ref2 = ref_features2['keypoints'][matches2[..., 0]]
            points_moving = moving_features['keypoints'][matches[..., 1]]
            # points_moving2 = moving_features2['keypoints'][matches2[..., 1]]

            num_matches1 = matches.shape[0]

            # Save image of corresponding matches
            axes = viz2d.plot_images([ref_image, moving_image])
            viz2d.plot_matches(points_ref, points_moving, color='lime', lw=0.2)
            viz2d.save_plot(self.save_dir.joinpath("keypoints", f"keypoints_{mov}_to_{ref}.png"))

            # Now do the same for the rotated image	
            moving_image = self.rotated_images[mov]
            center = (moving_image.shape[1]//2, moving_image.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 0, 1)
            moving_image = cv2.warpAffine(moving_image, rotation_matrix, moving_image.shape[:-1][::-1], borderValue=(255, 255, 255))
            
            moving_image = torch.tensor(moving_image.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
            moving_features = self.extractor.extract(moving_image)
            ref_features = self.extractor.extract(ref_image)

            matches01 = self.matcher({'image0': ref_features, 'image1': moving_features})
            ref_features, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
            matches = matches01['matches']
            points_ref = ref_features['keypoints'][matches[..., 0]]
            points_moving = moving_features['keypoints'][matches[..., 1]]

            num_matches2 = matches.shape[0]

            axes = viz2d.plot_images([ref_image, moving_image])
            viz2d.plot_matches(points_ref, points_moving, color='lime', lw=0.2)
            viz2d.save_plot(self.save_dir.joinpath("keypoints", f"keypoints_{mov}_to_{ref}_v2.png"))

            print(f"Num matches: {num_matches1} vs {num_matches2}")

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
            plt.savefig(self.save_dir.joinpath("warps", f"warped_{mov}.png"), dpi=300, bbox_inches="tight")
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