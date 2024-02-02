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
        help="Path to the data directory with images and masks."
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
    save_dir.mkdir(parents=True, exist_ok=True)

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
    for pt in patients[1:]:

        print(f"\nProcessing patient {pt.name}")

        constructor = Hiprova(
            data_dir = data_dir.joinpath(pt.name), 
            save_dir = save_dir.joinpath(pt.name),
            tform_tps = False,
        )
        constructor.run()

    return


if __name__ == "__main__":
    main()