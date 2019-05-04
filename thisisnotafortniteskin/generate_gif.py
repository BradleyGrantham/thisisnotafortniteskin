import glob
import os

import click
import imageio
from tqdm import tqdm

import thisisnotafortniteskin.constants

LOGGER = thisisnotafortniteskin.constants.logger


@click.command()
@click.argument("image-directory")
@click.argument("output-path")
def generate_gif(image_directory: str, output_path: str):
    paths = glob.glob(os.path.join(image_directory, "*.png"))
    images = []
    for path in tqdm(
        sorted(paths, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    ):
        images.append(imageio.imread(path))

    LOGGER.info("Saving GIF")
    imageio.mimsave(output_path, images)


if __name__ == "__main__":
    generate_gif()
