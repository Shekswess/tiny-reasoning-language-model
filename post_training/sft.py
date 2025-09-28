"""SFT Post-Training Script with EOS fix and dataset preview."""

import argparse
import multiprocessing

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

load_dotenv()  # Load environment variables from .env file


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="SFT Post-Training Script")
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the config file"
    )
    arguments = parser.parse_args()

    # Parsing the path to the config file
    config_path = arguments.config_path
    logger.info(f"Using config file at: {config_path}")

    # Load configuration from YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Config: {config}")

    # Check for GPU availability
    logger.info("GPU available: " + str(torch.cuda.is_available()))
    logger.info("GPU count: " + str(torch.cuda.device_count()))

    dtype = torch.bfloat16
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Using dtype: {dtype}")

    # Load model and tokenizer
    model_name = config["model"]
    tokenizer_name = config.get("tokenizer") or config["model"]
    logger.info(f"Loading model from: {model_name} and tokenizer from: {tokenizer_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Apply chat template if provided
    if config.get("chat_template"):
        tokenizer.chat_template = config["chat_template"]

    # Add any extra tokens
    extra_tokens = config.get("tokenizer_additional_special_tokens")
    if extra_tokens:
        extra_tokens = list(dict.fromkeys(extra_tokens))
        logger.info(f"Adding {len(extra_tokens)} special tokens to tokenizer")
        added_count = tokenizer.add_special_tokens(
            {"additional_special_tokens": extra_tokens}
        )
        if added_count:
            logger.info(
                "Resizing model embeddings from %s to %s",
                model.get_input_embeddings().num_embeddings,
                len(tokenizer),
            )
            model.resize_token_embeddings(len(tokenizer))

    # --- FIX EOS/PAD setup ---
    if "<|im_end|>" in tokenizer.get_vocab():
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|im_end|>"
        model.config.eos_token_id = eos_id
        model.config.pad_token_id = eos_id
        logger.info(f"Set eos_token_id={eos_id} for <|im_end|>")
    else:
        logger.warning("⚠️ <|im_end|> not found in tokenizer vocab!")

    tokenizer.padding_side = "right"

    # Load and preprocess the dataset
    data = load_dataset(
        config["dataset"]["name"],
        config["dataset"].get("subset"),
        split=config["dataset"]["split"],
        num_proc=config["dataset"].get("num_proc")
        if config["dataset"].get("num_proc") or config["dataset"].get("streaming")
        else multiprocessing.cpu_count(),
        streaming=config["dataset"].get("streaming"),
    )

    # --- Print one example to verify template/EOS works ---
    sample = next(iter(data))
    if "messages" in sample:
        logger.info("Previewing one tokenized example with chat template:")
        enc = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        logger.info("==== Chat Template Applied ====")
        logger.info(enc)
        logger.info("===============================")

    # Define Training Arguments and Trainer
    training_arguments = SFTConfig(**config["trainer"])
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=training_arguments,
    )

    trainer.train()
    logger.info("Training Done! 💥")
