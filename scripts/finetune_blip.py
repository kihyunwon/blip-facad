%%writefile finetune_blip.py
import logging
import sys
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import trl
import transformers
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


# custome trl.trainer.ConstantLengthDataset
class FashionImageCaptioningDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, size, processor):
        self.dataset = dataset
        self.size = size
        self.processor = processor
        
    def __len__(self):
        return self.size
    
    def __iter__(self):
        n = len(self.dataset['image'])
        m = len(self.dataset['text'])
        if n != m:
            raise Exception(f'Expects same image and text datasets, but received {n} images and {m} texts.')

        for i in range(n):
            image_iterator = iter(self.dataset['image'][i])
            text_iterator = iter(self.dataset['text'][i])
            while True:
                try:
                    image = next(image_iterator)['images']
                    text = next(text_iterator)['text']
                    example = self.processor(images=torch.tensor(image, dtype=torch.int), padding="max_length", return_tensors="pt")
                    example = {k: v.squeeze() for k, v in example.items()}
                    example['labels'] = text
                    yield example
                except Exception as err:
                    logger.warning(f"Error generating example: {err}")
                    break

@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    train_data_files: str
    train_data_size: int = 0
    eval_data_files: str = None
    eval_data_size: int = 0
    freeze_vision_model: bool = False
    freeze_text_model: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = True
    peft_target_model: Optional[str] = "blip-image-captioning-facad"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 16
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "blip-image-captioning-facad":
                self.peft_target_modules = [
                    "self.query",
                    "self.key",
                    "self.value",
                    "output.dense",
                    "self_attn.qkv",
                    "self_attn.projection",
                    "mlp.fc1",
                    "mlp.fc2",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    def from_pretrained_kwargs(self, training_args):
        kwargs = {}
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        if self.use_flash_attention_2:
            kwargs["attn_implementation"] = "flash_attention_2"
        return kwargs

def load_datasets(data_files):
    datasets = {'image': [], 'text': []}
    for data_file in data_files:
        dataset = None
        if data_file.endswith('.hdf5'):
            dataset = load_dataset("hdf5.py", name="keyed_config", key="images", data_files=data_file, trust_remote_code=True, streaming=True)
            datasets['image'].append(dataset['train'])
        else:
            dataset = load_dataset("text", data_files=data_file, streaming=True)
            datasets['text'].append(dataset['train'])
    return datasets

def main():
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    logger.info(f"Training parameters {training_args}\nSupervised Fine-Tuning parameters {sft_training_args}")

    processor = AutoProcessor.from_pretrained(sft_training_args.model_name_or_path)
    kwarg = sft_training_args.from_pretrained_kwargs(training_args)
    model = BlipForConditionalGeneration.from_pretrained(sft_training_args.model_name_or_path, **kwarg).to(device)

    peft_config = None
    if sft_training_args.use_peft:
        peft_config = LoraConfig(
            r=sft_training_args.peft_lora_r,
            target_modules=sft_training_args.peft_target_modules,
            lora_alpha=sft_training_args.peft_lora_alpha,
            lora_dropout=sft_training_args.peft_lora_dropout,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        if training_args.gradient_checkpointing:
            for param in model.parameters():
                param.requires_grad = False
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if sft_training_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if sft_training_args.freeze_text_model:
        _freeze_params(model.text_model)

    train_dataset = None
    if sft_training_args.train_data_files:
        data_files = sft_training_args.train_data_files.split(',')
        train_dataset = load_datasets(data_files)
        train_dataset = FashionImageCaptioningDataset(train_dataset, sft_training_args.train_data_size, processor)

    eval_dataset = None
    if sft_training_args.eval_data_files:
        data_files = sft_training_args.eval_data_files.split(',')
        eval_dataset = load_datasets(data_files)
        eval_dataset = FashionImageCaptioningDataset(eval_dataset, sft_training_args.eval_data_size, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=False, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    num_epochs = int(training_args.num_train_epochs)
    patience = 10
    min_eval_loss = float("inf")
    early_stopping_hook = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for _, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
            pixel_values = batch.pop('pixel_values').to(device)
            texts = batch.pop('labels')
            text_inputs = processor.tokenizer(texts, padding=True, return_tensors="pt").to(device)
            input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=input_ids)

            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        for _, batch in zip(tqdm(range(len(eval_dataloader)), desc='Validating batch: ...'), eval_dataloader):
            pixel_values = batch.pop('pixel_values').to(device)
            texts = batch.pop('labels')
            text_inputs = processor.tokenizer(texts, padding=True, return_tensors="pt").to(device)
            input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=input_ids)

            loss = outputs.loss
            eval_loss += loss.item()

        print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch+1, epoch_loss/len(train_dataloader), eval_loss/len(eval_dataloader), optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if eval_loss < min_eval_loss:
            model.save_pretrained(training_args.output_dir, from_pt=True)
            min_eval_loss = eval_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                break


if __name__ == "__main__":
    print(f'using transformers: {transformers.__version__}, trl: {trl.__version__}')
    main()