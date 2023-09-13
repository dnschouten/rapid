import pyvips
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import multiresolutionimageinterface as mir
from scipy import ndimage
from pathlib import Path

from hiprova import Hiprova


def collect_arguments():
    """
    Function to collect all arguments.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Parse arguments for 3D reconstruction.')
    parser.add_argument(
        "--datadir",
        required=True,
        type=Path,
        help="Path to the data directory."
    )
    parser.add_argument(
        "--savedir",
        required=True,
        type=Path,
        help="Path to the save directory."
    )
    args = parser.parse_args()

    data_dir = args.datadir
    save_dir = args.savedir

    assert data_dir.is_dir(), "Data directory does not exist."

    return data_dir, save_dir


def main(): 
    """
    Main function.
    """

    # Get args
    data_dir, save_dir = collect_arguments()
    
    # Get patients
    patients = sorted([i for i in data_dir.iterdir() if i.is_dir()])

    # Run 3D reconstruction
    for pt in patients:
        hiprova = Hiprova(
            data_dir = data_dir.joinpath(pt.name), 
            save_dir = save_dir.joinpath(pt.name)
        )
        hiprova.load_images()
        hiprova.load_masks()
        hiprova.get_contours()
        hiprova.get_rdp_contour()
        hiprova.get_rotations()
        hiprova.match_centerpoints()
        hiprova.match_rotations()
        # hiprova.finetune_reconstruction_icp()
        hiprova.finetune_reconstruction_lightglue()

    return

if __name__ == "__main__":
    main()