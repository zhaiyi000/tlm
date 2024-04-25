from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import os
import json
import tqdm


def train_tokenizer(files, save_path, test_length):
    tmp_file = f"{os.path.basename(os.path.abspath(__file__))}.text"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    tmp_file_f = open(tmp_file, 'a')
    for file in tqdm.tqdm(files):
        with open(file, 'r') as f:
            lines = f.read().strip().split("\n")
        for line in lines:
            try:
                text = json.loads(line)["text"]
            except:
                text = line
            tmp_file_f.write(text)
            tmp_file_f.write("\n")
    tmp_file_f.close()

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]", "[BOS]"])
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.train([tmp_file], trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")

    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer)
    tokenizer_fast.add_special_tokens({
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    })
    tokenizer_fast.save_pretrained(save_path)

    if test_length:
        test_model_max_length(tmp_file, save_path)

    if os.path.exists(tmp_file):
        os.remove(tmp_file)


def test_model_max_length(file, save_path):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=True
    )
    raw_datasets["train"] = load_dataset(
        extension,
        data_files=data_files,
        split=f"train",
        keep_in_memory=True
    )
    tokenizer = AutoTokenizer.from_pretrained(save_path)

    model_max_length = -1

    def tokenize_function(examples):
        encode = tokenizer(examples["text"])
        input_ids = encode["input_ids"]
        nonlocal model_max_length
        for ids in input_ids:
            model_max_length = max(model_max_length, len(ids))
        return encode

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    print("model_max_length:", model_max_length)
    tokenizer.model_max_length = ((model_max_length - 1) // 32 + 1) * 32
    tokenizer.save_pretrained(save_path)