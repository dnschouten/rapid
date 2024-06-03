import argparse
import pandas as pd
import json
import numpy as np
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
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        default="affine",
        help="Mode to run hiprova, options are ['prealignment', 'affine', 'deformable', 'valis', 'baseline']."
    )

    args = parser.parse_args()

    data_dir = args.datadir
    save_dir = args.savedir
    mode = args.mode.lower()

    assert data_dir.is_dir(), "Data directory does not exist."
    save_dir.mkdir(parents=True, exist_ok=True)
    assert mode in ["prealignment", "affine", "deformable", "valis", "baseline"], "Mode not recognized, must be any of ['prealignment', 'affine', 'deformable', 'valis', 'baseline']."

    return data_dir, save_dir, mode


def main(): 
    """
    Main function.
    """

    np.random.seed(42)

    # Get args
    data_dir, save_dir, mode = collect_arguments()
    
    # Get patients
    patients = sorted([i for i in data_dir.iterdir() if i.is_dir()])

    print(f"\nRunning job with following parameters:" \
          f"\n - data directory: {data_dir}" \
          f"\n - save directory: {save_dir}" \
          f"\n - mode: {mode}" \
          f"\n - num patients: {len(patients)}"
    )
    df = pd.DataFrame()

    for pt in patients:

        print(f"\nProcessing patient {pt.name}")

        constructor = Hiprova(
            data_dir = data_dir.joinpath(pt.name), 
            save_dir = save_dir.joinpath(pt.name),
            mode = mode,
        )
        constructor.run()

        # Save results in dataframe
        new_df = pd.DataFrame({
            "case": [pt.name],
            "dice": [constructor.reconstruction_dice],
            "tre": [constructor.tre],
            "sphericity": [constructor.sphericity],
            "median_contour_dist": [constructor.contour_distance],
            "mode": [mode],
        })
        df = pd.concat([df, new_df], ignore_index=True)

    # Save dataframe
    df.to_excel(save_dir.joinpath("aggregate_metrics.xlsx"), index=False)

    # Save most important config details
    config = {
        "mode": constructor.mode,
        "detector": constructor.detector_name,
        "keypoint_level": constructor.keypoint_level,
        "deformable_level": constructor.deformable_level,
        "evaluation_level": constructor.evaluation_level,
        "affine_ransac": constructor.affine_ransac,
        "deformable_ransac": constructor.deformable_ransac,
        "ransac_thresholds": constructor.ransac_thres_affine,
    } 
    with open(save_dir.joinpath("config.json"), "w") as f:
        json.dump(config, f)

    return


if __name__ == "__main__":
    main()