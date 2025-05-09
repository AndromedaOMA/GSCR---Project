import torch

class RoNACCDatasetPaired(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 128):
        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) % 2 == 0, "File should contain alternating incorrect and correct sentences."
        self.samples = []
        for i in range(0, len(lines), 2):
            self.samples.append((lines[i], 0))      # Correct
            self.samples.append((lines[i + 1], 1))  # incorrect
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }