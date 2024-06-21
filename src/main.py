import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path

from rapid import Rapid


def collect_arguments():
    """
    Function to collect all arguments.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Parse arguments for 3D reconstruction.')
    parser.add_argument(
        "--datapath",
        required=True,
        type=Path,
        help="Path to a csv/excel file containing the data to be reconstructed. This should be a 2-column file with the first " +
              "column containing the case name (i.e. case_01) and the second column containg the full absolute path to the image " + 
              "(i.e. /data/case_01/01.tif). If you only want to include every n-th image, you can just add the names of these images" + 
              "in this column and the algorithm will skip the other images."
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
        help="Mode to run RAPID, options are ['prealignment', 'affine', 'deformable', 'valis', 'baseline']."
    )

    args = parser.parse_args()

    data_path = args.datapath
    save_dir = args.savedir
    mode = args.mode.lower()

    assert data_path.exists(), "Data file does not exist."
    save_dir.mkdir(parents=True, exist_ok=True)
    assert mode in ["prealignment", "affine", "deformable", "valis", "baseline"], "Mode not recognized, must be any of ['prealignment', 'affine', 'deformable', 'valis', 'baseline']."

    return data_path, save_dir, mode


def main(): 
    """
    Main function.
    """

    np.random.seed(42)

    # Get args
    data_path, save_dir, mode = collect_arguments()
    
    # Load excel 
    data_path = Path(data_path)
    data_overview = pd.read_csv(data_path) if data_path.suffix == ".csv" else pd.read_excel(data_path)
    
    # Get cases
    cases = np.unique(list(data_overview["case"].values))
    files_per_case = [data_overview[data_overview["case"] == case].filename.tolist() for case in cases]

    print(f"\nRunning job with following parameters:" \
          f"\n - data path: {data_path}" \
          f"\n - save directory: {save_dir}" \
          f"\n - mode: {mode}" \
          f"\n - num cases: {len(cases)}"
    )
    df = pd.DataFrame()

    for case, files in zip(cases, files_per_case):

        print(f"\nProcessing case {case}")

        constructor = Rapid(
            case = case,
            files = files, 
            save_dir = save_dir.joinpath(case),
            mode = mode,
        )
        constructor.run()

        # Save results in dataframe
        new_df = pd.DataFrame({
            "case": [case],
            "dice": [constructor.reconstruction_dice],
            "tre": [constructor.tre],
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