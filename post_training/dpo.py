"""DPO Post-Training Script."""

import argparse
import multiprocessing

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

load_dotenv()  # Load environment variables from .env file


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="DPO Post-Training Script")
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

    # Load the model and tokenizer
    model_name = config["model"]
    tokenizer_name = config.get("tokenizer") or model_name

    logger.info(f"Loading policy model from: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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

    training_arguments = DPOConfig(**config["trainer"])

    trainer = DPOTrainer(
        model=model,
        args=training_arguments,
        train_dataset=data,
        processing_class=tokenizer,
    )

    trainer.train()
    logger.info("DPO Training Done! 💥")
