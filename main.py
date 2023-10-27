import matplotlib.pyplot as plt
import argparse
import multiresolutionimageinterface as mir
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
        "--maskdir",
        required=True,
        type=Path,
        help="Path to the masks directory."
    )
    parser.add_argument(
        "--savedir",
        required=True,
        type=Path,
        help="Path to the save directory."
    )

    args = parser.parse_args()

    data_dir = args.datadir
    mask_dir = args.maskdir
    save_dir = args.savedir

    assert data_dir.is_dir(), "Data directory does not exist."
    assert mask_dir.is_dir(), "Mask directory does not exist."

    return data_dir, mask_dir, save_dir


def main(): 
    """
    Main function.
    """

    # Get args
    data_dir, mask_dir, save_dir = collect_arguments()
    
    # Get patients
    patients = sorted([i for i in data_dir.iterdir() if i.is_dir()])

    # Run 3D reconstruction
    for pt in patients:
        hiprova = Hiprova(
            data_dir = data_dir.joinpath(pt.name), 
            save_dir = save_dir.joinpath(pt.name),
            mask_dir = mask_dir.joinpath(pt.name),
            detector = "DALF"
        )
        hiprova.load_images()
        hiprova.load_masks()
        hiprova.apply_masks()
        hiprova.find_rotations()
        hiprova.prealignment()
        hiprova.finetune_reconstruction()
        hiprova.create_3d_volume()
        hiprova.interpolate_3d_volume()
        hiprova.plot_3d_volume()

    return


if __name__ == "__main__":
    main()