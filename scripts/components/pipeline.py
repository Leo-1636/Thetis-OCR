from datasets import concatenate_datasets, load_dataset

from components.config import DataConfig

COLUMNS = ["image", "question", "answer"]

def to_message(sample):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": sample["question"]},
            {"type": "image", "image": sample["image"]},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]}

class Dataset:
    @classmethod
    def load_datasets(cls, configs: list[DataConfig], split: str):
        if not configs:
            return None
        return concatenate_datasets([cls.load_dataset(config, split) for config in configs])

    @classmethod
    def load_dataset(cls, config: DataConfig, split: str):
        if config.samples:
            split = f"{split}[:{config.samples}]"
        dataset = load_dataset(config.name, split = split)

        if "question" not in dataset.column_names:
            if not config.prompt:
                raise ValueError(f"{config.name} 沒有 question 欄位，請在 data.yaml 設定 prompt")
            dataset = dataset.add_column("question", [config.prompt] * len(dataset))

        return dataset.select_columns(COLUMNS)
