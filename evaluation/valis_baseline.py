import argparse
import pathlib
import tqdm
from valis import registration, slide_io, affine_optimizer, feature_detectors
from pathlib import Path


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

        return


    def register(self):
        """
        Method to register slides using Valis.
        """

        # Perform rigid + non-rigid registration
        registrar = registration.Valis(
            self.datadir,
            self.savedir,
            imgs_ordered=True,
        )
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        # Optional micro registration
        if self.perform_micro:
            registrar.register_micro(
                max_non_rigid_registration_dim_px=2000,
                align_to_reference=False
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

        return


def main():
    """
    Run registration
    """

    # Get all cases that have not been registered yet
    datadir, savedir, micro, fullres = collect_arguments()
    cases = sorted(list(datadir.iterdir()))
    cases = [i for i in cases if not savedir.joinpath(i.name).is_dir()]

    print(f"Found {len(cases)} patients to register")

    # Run registration
    for case in cases:
        reg = Registration(
            datadir=case,
            savedir=savedir,
            micro=micro,
            fullres=fullres
        )
        reg.register()
        reg.convert_ometiff()

    return


if __name__ == "__main__":
    main()
