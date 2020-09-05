from omegaconf import DictConfig

from algorithms.modules_cswm import EncoderCNNLarge


def train(config: DictConfig):
    print(config.pretty())

    print(EncoderCNNLarge)

    return 123
