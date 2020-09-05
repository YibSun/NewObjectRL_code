import hydra

from omegaconf import DictConfig


@hydra.main(config_path='../Config/defaults_data.yaml')
def main(config: DictConfig):
    print(config.pretty())

if __name__ == '__main__':
    main()
