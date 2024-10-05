import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print entire configuration (useful for debugging)
    # print(OmegaConf.to_yaml(cfg))
    print(cfg.keys())
    # Access configuration values like this
    print(f"Training model: {cfg.pretrain.model.arch}")
    print(f"Resolution: {cfg.pretrain.validation.resolution}")
    print(f"Dataset: {cfg.dataloader.train_dataset}")
    
    # Run training, validation, etc.
    if cfg.pretrain.training.eval_only:
        print("Evaluation only mode")
    else:
        print(f"Training for {cfg.pretrain.training.epochs} epochs")

if __name__ == "__main__":
    main()
