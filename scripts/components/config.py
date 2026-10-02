from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

RootPath = Path.cwd()

class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ModelConfig(BaseConfig):
    name: str = Field(..., description="Model name")
    load_in_4bit: bool = Field(..., description="Load in 4bit")
    chat_template: str | None = Field(..., description="Chat template")

class LoRAConfig(BaseConfig):
    r: int = Field(..., description="Rank")
    lora_alpha: int = Field(..., description="Lora alpha")
    lora_dropout: float = Field(..., description="Lora dropout")

    use_rslora: bool = Field(..., description="Use rslora")
    random_state: int = Field(..., description="Random state")

class SFTConfig(BaseConfig):
    epochs: int = Field(..., description="Epochs")
    batch_size: int = Field(..., description="Batch size")
    gradient_steps: int = Field(..., description="Gradient steps")

    warmup_ratio: float = Field(..., description="Warmup ratio")
    max_grad_norm: float = Field(..., description="Max grad norm")

    learning_rate: float = Field(..., description="Learning rate")
    lr_scheduler_type: str = Field(..., description="Lr scheduler type")

    seed: int = Field(..., description="Seed")
    optim: str = Field(..., description="Optimizer")    
    weight_decay: float = Field(..., description="Weight decay")

    steps: int = Field(..., description="Steps")
    strategy: str = Field(..., description="Strategy")
    output_dir: str = Field(..., description="Output directory")


class TrainConfig(BaseConfig):
    model: ModelConfig
    lora: LoRAConfig
    sft: SFTConfig

    @classmethod
    def load_config(cls, path: str | Path):
        try:
            return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {e}")

class DataConfig(BaseConfig):
    name: str = Field(..., description="Dataset name")
    prompt: str | None = Field(None, description="Prompt")
    samples: int | None = Field(None, description="Samples")

class DatasetConfig(BaseConfig):
    train: list[DataConfig] = Field([], description="Train datasets")
    test: list[DataConfig] = Field([], description="Test datasets")

    @classmethod
    def load_config(cls, path: str | Path):
        try:
            return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {e}")

# ============ 評估：configs/test.yaml（之後加） ============
# class EvalConfig(Config):
#     track: TrackConfig
#     ...
