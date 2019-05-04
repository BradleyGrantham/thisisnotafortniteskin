import click
import torch.onnx

from thisisnotafortniteskin.networks import Generator


@click.command()
@click.argument("pt-model-path")
def convert_to_onnx(pt_model_path):
    path_to_onnx_model = ("/".join(pt_model_path.split("/")[:-1]) + "/"
                          + pt_model_path.split("/")[-1].split(".")[0]
                          + ".onnx")
    model = Generator()
    model.load_state_dict(torch.load(pt_model_path, map_location='cpu'))

    dummy_input = torch.randn(1, 100, 1, 1)

    output_names = ["skin"]

    torch.onnx.export(model, dummy_input, path_to_onnx_model,
                      output_names=output_names)


if __name__ == "__main__":
    convert_to_onnx()
