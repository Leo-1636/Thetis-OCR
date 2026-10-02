import argparse

from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator

from trl import SFTTrainer, SFTConfig

from components.config import TrainConfig, DatasetConfig
from components.pipeline import Dataset, to_message

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type = str,
    default = "configs/train.yaml",
    help = "Path to the training config (default: configs/train.yaml)",
)
parser.add_argument(
    "--datasets",
    type = str,
    default = "configs/data.yaml",
    help = "Path to the dataset config (default: configs/data.yaml)",
)
args = parser.parse_args()
config = TrainConfig.load_config(args.config)
data_config = DatasetConfig.load_config(args.datasets)

model, tokenizer = FastVisionModel.from_pretrained(
    config.model.name,
    load_in_4bit = config.model.load_in_4bit,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = config.lora.r,
    lora_alpha = config.lora.lora_alpha,
    lora_dropout = config.lora.lora_dropout,

    use_rslora = config.lora.use_rslora,
    random_state = config.lora.random_state,
)
FastVisionModel.for_training(model)

if config.model.chat_template:
    tokenizer = get_chat_template(
        tokenizer,
        config.model.chat_template,
    )

train_dataset = Dataset.load_datasets(data_config.train, "train")
eval_dataset  = Dataset.load_datasets(data_config.test, "test")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer, formatting_func = to_message),

    train_dataset = train_dataset,
    eval_dataset = eval_dataset,

    args = SFTConfig(
        per_device_train_batch_size = config.sft.batch_size,
        per_device_eval_batch_size  = config.sft.batch_size,
        gradient_accumulation_steps = config.sft.gradient_steps,

        num_train_epochs = config.sft.epochs,
        warmup_ratio     = config.sft.warmup_ratio,
        max_grad_norm    = config.sft.max_grad_norm,

        learning_rate     = config.sft.learning_rate,
        lr_scheduler_type = config.sft.lr_scheduler_type,

        seed  = config.sft.seed,
        optim = config.sft.optim,
        weight_decay = config.sft.weight_decay,
        
        logging_steps = 1,
        output_dir = config.sft.output_dir,

        save_steps    = config.sft.steps,
        eval_steps    = 0.1,
        save_strategy = config.sft.strategy,
        eval_strategy = config.sft.strategy,

        report_to = "wandb",

        # For multi-GPU (DDP):
        ddp_find_unused_parameters = False,

        # For vision finetuning:
        max_length = 2048,
        dataset_kwargs = {
            "skip_prepare_dataset": True,
        },
        dataset_text_field = "",
        remove_unused_columns = False,
    )
)

trainer.train()

if trainer.is_world_process_zero():     # 多卡時只讓主 process 寫檔
    model.save_pretrained(config.sft.output_dir)
    tokenizer.save_pretrained(config.sft.output_dir)
