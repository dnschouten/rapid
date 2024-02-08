import argparse
import pathlib
import tqdm
import numpy as np
import subprocess
import shutil
from valis import registration, slide_io, affine_optimizer, feature_detectors
from pathlib import Path
from valis.micro_rigid_registrar import MicroRigidRegistrar # For high resolution rigid registration


def collect_arguments():
    """
    Function to collect the arguments
    """

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Register slides using valis"
    )
    parser.add_argument(
        "--datadir",
        required=True,
        type=pathlib.Path,
        help="Path to the cases to register"
    )
    parser.add_argument(
        "--savedir",
        required=True,
        type=pathlib.Path,
        help="Directory to save the results",
    )
    parser.add_argument(
        "--micro",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to perform micro registration",
    )
    parser.add_argument(
        "--fullres",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to save fullres results",
    )
    args = parser.parse_args()

    datadir = pathlib.Path(args.datadir)
    savedir = pathlib.Path(args.savedir)
    micro = args.micro
    fullres = args.fullres

    assert datadir.is_dir(), f"provided directory {datadir} does not exist"
    if not savedir.is_dir():
        savedir.mkdir(parents=True)
    assert micro in [True, False], "something fishy going on with micro arg"
    assert fullres in [True, False], "something fishy going on with fullres arg"

    return datadir, savedir, micro, fullres


class Registration:

    def __init__(self, datadir, savedir, micro, fullres):

        self.datadir = datadir
        self.savedir = savedir
        self.perform_micro = micro
        self.perform_fullres = fullres
        self.case = self.datadir.name

        assert datadir.exists(), f"datadir {datadir} does not exist"

        print(f"\nRegistering case {self.case}")

        return


    def copy_to_local(self):
        """
        Copy the slides to the Docker container.
        """

        print(f" - copying slides to local")

        # Copy slides locally
        self.local_data_dir = Path(f"/tmp/valis/input/{self.datadir.name}")
        self.local_data_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.call(f"cp -r {self.datadir} {self.local_data_dir}", shell=True)

        self.local_save_dir = Path(f"/tmp/valis/output/{self.datadir.name}")
        self.local_save_dir.mkdir(parents=True, exist_ok=True)

        return

    def register(self):
        """
        Method to register slides using Valis.
        """

        print(f" - performing registration")

        # Perform rigid + non-rigid registration
        registrar = registration.Valis(
            str(self.local_data_dir),
            str(self.local_save_dir),
            imgs_ordered=True
        )
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        # Optional micro registration
        if self.perform_micro:
            # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
            micro_reg_fraction = 0.25

            img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
            min_max_size = np.min([np.max(d) for d in img_dims])
            img_areas = [np.multiply(*d) for d in img_dims]
            max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
            micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)

            micro_reg, micro_error = registrar.register_micro(
                max_non_rigid_registration_dim_px=micro_reg_size,
            )

        # Save results
        if self.perform_fullres:
            self.save_slide_path = self.savedir.joinpath(self.case, "registered_slides")
            registrar.warp_and_save_slides(str(self.save_slide_path), compression="jpeg")

        return


    def convert_ometiff(self):
        """
        Method to convert the ome tiff files to tif
        """

        if self.perform_fullres:

            # Identify files
            slides = sorted(list(self.save_slide_path.iterdir()))

            # Convert files
            print(f"converting {len(slides)} slides to .tif")
            for slide in tqdm.tqdm(slides):

                # Get reader for slide format
                reader_cls = slide_io.get_slide_reader(slide, series=0)
                reader = reader_cls(slide, series=0)

                # Extract image
                img = reader.slide2vips(level=0)

                # Save as pyvips
                savepath = self.save_slide_path.joinpath(f"{slide.name.split('.')[0]}.tif")
                img.write_to_file(
                    savepath,
                    tile=True,
                    compression="jpeg",
                    Q=20,
                    pyramid=True
                )

        registration.kill_jvm()

        return

    def copy_to_remote(self):
        """
        Copy the results to the remote directory
        """

        print(f" - copying slides to remote")

        # Copy slides locally
        subprocess.call(f"cp -r {self.local_save_dir} {self.savedir}", shell=True)
        shutil.rmtree(self.local_data_dir)
        shutil.rmtree(self.local_save_dir)

        return


def main():
    """
    Run registration
    """

    # Get all cases that have not been registered yet
    datadir, savedir, micro, fullres = collect_arguments()
    cases = sorted(list(datadir.iterdir()))

    print(f"\nFound {len(cases)} patients to register")

    # Run registration
    for case in cases:
        reg = Registration(
            datadir=case,
            savedir=savedir.joinpath(case.name),
            micro=micro,
            fullres=fullres
        )
        reg.copy_to_local()
        reg.register()
        reg.convert_ometiff()
        reg.copy_to_remote()

    return


if __name__ == "__main__":
    main()
