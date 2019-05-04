import click
import matplotlib.pyplot as plt
import torch
import torchvision.transforms

import thisisnotafortniteskin.constants as constants
from thisisnotafortniteskin.networks import Generator


def load_generator(path):
    model = Generator()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def generate_random(pt_model_path, show_plot=True, save_path=None, return_PIL=False):
    model = load_generator(pt_model_path)
    z = torch.randn(128, 100, 1, 1, device=constants.TORCH_DEVICE)
    with torch.no_grad():
        skin = model(z).detach().cpu()

    pil_image = torchvision.transforms.ToPILImage()(skin[0, :, :, :])

    plt.imshow(pil_image)
    if save_path is not None:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    if return_PIL:
        return pil_image


@click.command()
@click.argument("pt-model-path")
@click.option("--show-plot/--dont-show-plot", default=True)
@click.option("--save-path", default=None)
@click.option("--return-pil/--dont-return-pil", default=False)
def cli_generate_random(pt_model_path, show_plot=True, save_path=None):
    generate_random(pt_model_path, show_plot, save_path)


if __name__ == "__main__":
    cli_generate_random()
