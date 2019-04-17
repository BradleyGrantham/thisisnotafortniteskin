import glob
import os

import albumentations
import albumentations.augmentations.transforms as augmentations
import click
import cv2
from tqdm import tqdm

import thisisnotafortniteskin.constants

LOGGER = thisisnotafortniteskin.constants.logger


def random_rgb_shift(image):
    shift = albumentations.Compose(
        [
            augmentations.RGBShift(
                r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=1
            )
        ],
        p=1,
    )

    return shift(image=image)["image"]


def horizontal_flip(image):
    flip = augmentations.HorizontalFlip(p=1)
    return flip(image=image)["image"]


@click.command()
@click.argument("raw-image-directory")
@click.argument("augmented-save-directory")
@click.option("--number-of-rgb-shifts", default=40)
def perform_augmentations(raw_image_directory, augmented_save_directory,
                          number_of_rgb_shifts=40):
    LOGGER.info("Finding images")
    image_paths = glob.glob(os.path.join(raw_image_directory, "*"))

    LOGGER.info("Performing 1 horizontal flip and {} random RGB shifts for each image"
                .format(number_of_rgb_shifts))
    for image_path in tqdm(image_paths):
        image_name, image_ext = image_path.split("/")[-1].split(".")
        image = cv2.imread(image_path)
        cv2.imwrite(
            os.path.join(
                augmented_save_directory, (image_name + "_original" + "." + image_ext)
            ),
            image,
        )

        flip = horizontal_flip(image)
        cv2.imwrite(
            os.path.join(
                augmented_save_directory, (image_name + "_flip" + "." + image_ext)
            ),
            flip,
        )

        for i in range(number_of_rgb_shifts):
            shift = random_rgb_shift(image)
            cv2.imwrite(
                os.path.join(
                    augmented_save_directory,
                    (image_name + "_shift" + str(i) + "." + image_ext),
                ),
                shift,
            )


if __name__ == "__main__":
    perform_augmentations()
