import wandb
import hydra
import numpy as np
import torch

import sys
from omegaconf import OmegaConf, DictConfig
import datetime

from algorithms.train import train


# a helper function
def to_dotlist(config: DictConfig):
    path2value = {}

    def get_path(node, path):
        if not isinstance(node, DictConfig):
            path2value['.'.join(path)] = node  # OmegaConf.to_container(node)
        else:
            for key in node.keys():
                get_path(node[key], path + [key])

    get_path(config, [])
    return path2value


# main entrance with the parameter dictionary `config`
def main(config):
    print(config.algo.batch_size)  # TODO

    # for epoch in range(1000):
    #     loss = np.random.rand()
    #
    #     # log some metrics during training
    #     wandb.log({'epoch': epoch, 'loss': loss})
    #
    # # log some metrics after training
    # wandb.log({'test_metric': np.random.rand()})

    results = train(config)
    print('results', results)


# load YAML parameters and command line arguments (include W&B's sweeping hyperparameters)
@hydra.main(config_path='./Config/defaults.yaml')
def run(config: DictConfig):
    print(config.pretty())

    # init for W&B (only if you run `wandb login <token>` before)
    wandb.init(
        project='TestRun1',
        name='TestRun-' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        config=to_dotlist(config)  # convert a dictionary to strings ("dot-list") for logging in W&B
    )

    # TODO seeding
    np.random.seed(config.algo.seed)
    torch.manual_seed(config.algo.seed)
    if config.algo.cuda:
        torch.cuda.manual_seed(config.algo.seed)

    # call the main function
    main(config=config)


if __name__ == '__main__':
    # remove '--' before each argument `--XXX` for compatible between Hydra and W&B
    sys.argv[1:] = [arg[2:] if arg.startswith('--') else arg for arg in sys.argv[1:]]
    # run the entrance function
    run()
