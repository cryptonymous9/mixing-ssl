import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from modules.test import hello

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hello(cfg)
    print(cfg.distributed.use_submitit)
    # Print entire configuration (useful for debugging)
    # print(OmegaConf.to_yaml(cfg))

    # # Access configuration values like this
    # print(f"Pretraining: {cfg.pretrain.model.name}")
    # print(f"Training model: {cfg.pretrain.model.arch}")
    # print(f"Resolution: {cfg.pretrain.validation.resolution}")
    # print(f"Dataset: {cfg.data.train_dataset}")
    
    # # Run training, validation, etc.
    # if cfg.pretrain.training.eval_only:
    #     print("Evaluation only mode")
    # else:
    #     print(f"Training for {cfg.pretrain.training.epochs} epochs")

if __name__ == "__main__":
    main()
