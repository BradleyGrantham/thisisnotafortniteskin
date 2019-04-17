import logging

import torch

log_format = (
        "[%(asctime)s] | " "[%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    )
logging.basicConfig(format=log_format)
logger = logging.getLogger("thisisnotafortniteskin")
logger.setLevel(level=logging.INFO)

TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

