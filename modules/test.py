import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# @hydra.main(config_path="../configs/", config_name="config")
def hello(cfg: DictConfig):
    # Print entire configuration (useful for debugging)
    # print(OmegaConf.to_yaml(cfg))
    print(OmegaConf.to_yaml(cfg))

    # Access configuration values like this
    print(f"Pretraining: {cfg.pretrain.model.name}")
    print(f"Training model: {cfg.pretrain.model.arch}")
    print(f"Resolution: {cfg.pretrain.validation.resolution}")
    print(f"Dataset: {cfg.data.train_dataset}")
    print(cfg.pretrain.training.distributed)
    # Run training, validation, etc.
    if cfg.pretrain.training.eval_only:
        print("Evaluation only mode")
    else:
        print(f"Training for {cfg.pretrain.training.epochs} epochs")