"""Data preparation script to create a dataset for training using streaming."""

import argparse
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime
from itertools import islice
from typing import Any, Dict, Iterator, Optional

import psutil
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from loguru import logger


def clean_example(
    example: Dict[str, Any],
    columns_to_drop: Optional[list[str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Clean an example by removing and renaming configured columns.

    Args:
        example: Dictionary containing the example data
        columns_to_drop: Optional list of columns to remove if present
        rename_map: Optional mapping of column names to rename

    Returns:
        Cleaned example dictionary
    """
    cleaned_example = example.copy()

    if columns_to_drop:
        for column in columns_to_drop:
            if column in cleaned_example:
                del cleaned_example[column]
                logged_drops = getattr(clean_example, "_logged_drop_columns", set())
                if column not in logged_drops:
                    logger.info(f"Dropping column '{column}' as configured")
                    logged_drops.add(column)
                    clean_example._logged_drop_columns = logged_drops

    if rename_map:
        for old_name, new_name in rename_map.items():
            if old_name in cleaned_example:
                if new_name in cleaned_example:
                    logger.warning(
                        "Cannot rename column '{}' to '{}' because the target already exists",
                        old_name,
                        new_name,
                    )
                    continue
                cleaned_example[new_name] = cleaned_example.pop(old_name)
                logged_renames = getattr(
                    clean_example, "_logged_renamed_columns", set()
                )
                rename_key = f"{old_name}->{new_name}"
                if rename_key not in logged_renames:
                    logger.info(
                        f"Renaming column '{old_name}' to '{new_name}' as configured"
                    )
                    logged_renames.add(rename_key)
                    clean_example._logged_renamed_columns = logged_renames

    return cleaned_example


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the parsed configuration
    """
    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    logger.info(f"Loaded config for dataset: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Number of datasets to process: {len(config['datasets'])}")

    return config


def generate_dataset_metadata(
    config: Dict[str, Any], actual_counts: Dict[str, int]
) -> Dict[str, Any]:
    """Generate comprehensive metadata for the final dataset.

    Args:
        config: Configuration dictionary
        actual_counts: Dictionary mapping dataset names to actual entry counts

    Returns:
        Dictionary containing dataset metadata
    """
    total_entries = sum(actual_counts.values())

    dataset_sources = []
    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        subset = dataset_config.get("subset")
        split = dataset_config["split"]
        requested_entries = dataset_config["entries"]
        actual_entries = actual_counts.get(f"{dataset_name}_{subset}_{split}", 0)

        source_info = {
            "name": dataset_name,
            "subset": subset,
            "split": split,
            "requested_entries": requested_entries,
            "actual_entries": actual_entries,
            "percentage_of_total": round((actual_entries / total_entries) * 100, 2)
            if total_entries > 0
            else 0,
        }
        dataset_sources.append(source_info)

    metadata = {
        "name": config["name"],
        "description": config["description"],
        "total_entries": total_entries,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_datasets_count": len(config["datasets"]),
        "dataset_sources": dataset_sources,
        "composition_summary": f"This dataset contains {total_entries:,} entries from {len(config['datasets'])} source datasets.",
    }

    return metadata


def create_streaming_dataset_generator(config: Dict[str, Any]) -> Iterator[Dict]:
    """Create a generator that yields examples from all datasets in
    streaming mode.

    Args:
        config: Configuration dictionary containing dataset specifications

    Yields:
        Individual examples from the datasets
    """
    total_entries = 0

    logger.info("Starting to process datasets in streaming mode...")

    for i, dataset_config in enumerate(config["datasets"]):
        dataset_name = dataset_config["name"]
        subset = dataset_config.get("subset")
        split = dataset_config["split"]
        entries = dataset_config["entries"]
        columns_to_drop = dataset_config.get("drop_columns")
        rename_map = dataset_config.get("rename_columns")

        logger.info(
            f"Processing dataset {i + 1}/{len(config['datasets'])}: {dataset_name}"
        )
        logger.info(f"  Subset: {subset}, Split: {split}, Entries: {entries}")

        try:
            if subset:
                streaming_dataset = load_dataset(
                    dataset_name, subset, split=split, streaming=True
                )
            else:
                streaming_dataset = load_dataset(
                    dataset_name, split=split, streaming=True
                )

            if entries:
                dataset_iter = islice(streaming_dataset, entries)
                logger.info(f"  Processing {entries} entries")
            else:
                dataset_iter = streaming_dataset
                logger.info("  Processing all available entries")

            dataset_entries = 0
            for example in dataset_iter:
                cleaned_example = clean_example(
                    example, columns_to_drop=columns_to_drop, rename_map=rename_map
                )
                yield cleaned_example
                dataset_entries += 1
                total_entries += 1

                if total_entries % 10000 == 0:
                    logger.info(f"  Processed {total_entries} total entries so far...")

            logger.info(f"  Completed: {dataset_entries} entries from {dataset_name}")

        except Exception as e:
            logger.error(f"Failed to process dataset {dataset_name}: {str(e)}")
            raise

    logger.info(f"Streaming processing complete: {total_entries} total entries")


def create_dataset_from_generator(
    config: Dict[str, Any], batch_size: int = 1000
) -> tuple[Dataset, Dict[str, Any]]:
    """Create a dataset from the streaming generator by collecting
    examples in batches.

    Args:
        config: Configuration dictionary
        batch_size: Number of examples to collect in each batch

    Returns:
        Tuple of (Final concatenated dataset, metadata dictionary)
    """
    logger.info("Creating dataset from streaming data with chunked processing...")

    temp_dir = tempfile.mkdtemp(prefix="dataset_chunks_")
    chunk_files = []
    actual_counts = {}
    total_entries = 0
    chunk_idx = 0
    current_chunk = []

    try:
        for i, dataset_config in enumerate(config["datasets"]):
            dataset_name = dataset_config["name"]
            subset = dataset_config.get("subset")
            split = dataset_config["split"]
            entries = dataset_config["entries"]

            dataset_key = f"{dataset_name}_{subset}_{split}"
            columns_to_drop = dataset_config.get("drop_columns")
            rename_map = dataset_config.get("rename_columns")

            logger.info(
                f"Processing dataset {i + 1}/{len(config['datasets'])}: {dataset_name}"
            )

            try:
                if subset:
                    streaming_dataset = load_dataset(
                        dataset_name, subset, split=split, streaming=True
                    )
                else:
                    streaming_dataset = load_dataset(
                        dataset_name, split=split, streaming=True
                    )

                if entries:
                    dataset_iter = islice(streaming_dataset, entries)
                else:
                    dataset_iter = streaming_dataset

                dataset_entries = 0

                for example in dataset_iter:
                    cleaned_example = clean_example(
                        example,
                        columns_to_drop=columns_to_drop,
                        rename_map=rename_map,
                    )
                    example_with_source = cleaned_example.copy()
                    example_with_source["_source_dataset"] = dataset_name
                    example_with_source["_source_subset"] = subset
                    example_with_source["_source_split"] = split

                    current_chunk.append(example_with_source)
                    dataset_entries += 1
                    total_entries += 1

                    if len(current_chunk) >= batch_size:
                        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.json")
                        with open(chunk_path, "w") as f:
                            import json

                            json.dump(current_chunk, f)
                        chunk_files.append(chunk_path)
                        logger.info(
                            f"Saved chunk {chunk_idx} with {len(current_chunk)} examples"
                        )
                        current_chunk = []
                        chunk_idx += 1

                actual_counts[dataset_key] = dataset_entries
                logger.info(
                    f"  Completed: {dataset_entries} entries from {dataset_name}"
                )

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_name}: {str(e)}")
                raise

        if current_chunk:
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.json")
            with open(chunk_path, "w") as f:
                import json

                json.dump(current_chunk, f)
            chunk_files.append(chunk_path)
            logger.info(
                f"Saved final chunk {chunk_idx} with {len(current_chunk)} examples"
            )

        logger.info(
            f"Processed {total_entries} total examples in {len(chunk_files)} chunks"
        )

        logger.info("Creating dataset from chunks...")
        datasets_to_concatenate = []

        for i, chunk_path in enumerate(chunk_files):
            logger.info(f"Loading chunk {i + 1}/{len(chunk_files)}...")
            with open(chunk_path, "r") as f:
                import json

                chunk_data = json.load(f)

            if chunk_data:
                chunk_dataset = Dataset.from_list(chunk_data)
                datasets_to_concatenate.append(chunk_dataset)

            del chunk_data

        if datasets_to_concatenate:
            from datasets import concatenate_datasets

            logger.info(
                f"Concatenating {len(datasets_to_concatenate)} chunk datasets..."
            )
            dataset = concatenate_datasets(datasets_to_concatenate)
            logger.info("Concatenation complete")

            del datasets_to_concatenate

            logger.info("Shuffling dataset...")
            dataset = dataset.shuffle(seed=42)
            logger.info("Shuffling complete")
            metadata = generate_dataset_metadata(config, actual_counts)

            return dataset, metadata
        else:
            raise ValueError("No examples were collected from the datasets")

    finally:
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_dataset_memory_efficient(
    config: Dict[str, Any], output_path: str, chunk_size: int = 5000
) -> tuple[Dataset, Dict[str, Any]]:
    """Create dataset using memory-efficient Arrow file streaming.

    This approach writes data directly to Arrow format on disk,
    avoiding loading everything into memory.

    Args:
        config: Configuration dictionary
        output_path: Path where to save the Arrow file
        chunk_size: Number of examples per chunk

    Returns:
        Tuple of (Dataset loaded from Arrow file, metadata dictionary)
    """
    logger.info("Creating dataset using memory-efficient Arrow streaming...")

    temp_dir = tempfile.mkdtemp(prefix="arrow_chunks_")
    parquet_files = []
    actual_counts = {}
    total_entries = 0
    chunk_idx = 0
    current_chunk = []

    try:
        for i, dataset_config in enumerate(config["datasets"]):
            dataset_name = dataset_config["name"]
            subset = dataset_config.get("subset")
            split = dataset_config["split"]
            entries = dataset_config["entries"]

            dataset_key = f"{dataset_name}_{subset}_{split}"
            columns_to_drop = dataset_config.get("drop_columns")
            rename_map = dataset_config.get("rename_columns")

            logger.info(
                f"Processing dataset {i + 1}/{len(config['datasets'])}: {dataset_name}"
            )

            try:
                if subset:
                    streaming_dataset = load_dataset(
                        dataset_name, subset, split=split, streaming=True
                    )
                else:
                    streaming_dataset = load_dataset(
                        dataset_name, split=split, streaming=True
                    )

                if entries:
                    dataset_iter = islice(streaming_dataset, entries)
                else:
                    dataset_iter = streaming_dataset

                dataset_entries = 0

                for example in dataset_iter:
                    cleaned_example = clean_example(
                        example,
                        columns_to_drop=columns_to_drop,
                        rename_map=rename_map,
                    )
                    example_with_source = cleaned_example.copy()
                    example_with_source["_source_dataset"] = dataset_name
                    example_with_source["_source_subset"] = subset
                    example_with_source["_source_split"] = split

                    current_chunk.append(example_with_source)
                    dataset_entries += 1
                    total_entries += 1

                    if len(current_chunk) >= chunk_size:
                        parquet_path = os.path.join(
                            temp_dir, f"chunk_{chunk_idx}.parquet"
                        )

                        chunk_dataset = Dataset.from_list(current_chunk)
                        chunk_dataset.to_parquet(parquet_path)
                        parquet_files.append(parquet_path)

                        logger.info(
                            f"Saved parquet chunk {chunk_idx} with "
                            f"{len(current_chunk)} examples"
                        )

                        current_chunk = []
                        del chunk_dataset
                        chunk_idx += 1

                actual_counts[dataset_key] = dataset_entries
                logger.info(
                    f"  Completed: {dataset_entries} entries from {dataset_name}"
                )

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_name}: {str(e)}")
                raise

        if current_chunk:
            parquet_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.parquet")
            chunk_dataset = Dataset.from_list(current_chunk)
            chunk_dataset.to_parquet(parquet_path)
            parquet_files.append(parquet_path)
            logger.info(
                f"Saved final parquet chunk {chunk_idx} with "
                f"{len(current_chunk)} examples"
            )
            del chunk_dataset

        logger.info(
            f"Processed {total_entries} total examples in "
            f"{len(parquet_files)} parquet chunks"
        )

        logger.info("Loading dataset from parquet chunks...")
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")

        logger.info("Shuffling dataset...")
        dataset = dataset.shuffle(seed=42)
        logger.info("Shuffling complete")

        metadata = generate_dataset_metadata(config, actual_counts)

        return dataset, metadata

    finally:
        logger.info("Cleaning up temporary parquet files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def save_dataset_streaming(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    upload_to_hub: bool = False,
    batch_size: int = 5000,  # Increased default for better efficiency
    use_parquet_optimization: bool = True,  # New option for large datasets
):
    """Save the dataset by streaming directly to disk or hub without
    loading all in memory.

    Args:
        config: Configuration dictionary
        output_dir: Local directory to save the dataset (optional)
        upload_to_hub: Whether to upload to HuggingFace Hub
        batch_size: Batch size for processing (increased default)
        use_parquet_optimization: Use Arrow/Parquet for memory efficiency
    """
    if not output_dir and not upload_to_hub:
        raise ValueError("Must specify either output_dir or upload_to_hub")

    total_requested = sum(d["entries"] for d in config["datasets"])

    if use_parquet_optimization and total_requested > 50000:
        logger.info(
            f"Large dataset detected ({total_requested:,} entries). "
            f"Using memory-efficient Arrow processing..."
        )

        if output_dir:
            temp_output = os.path.join(output_dir, "temp_dataset")
            os.makedirs(temp_output, exist_ok=True)
            dataset, metadata = create_dataset_memory_efficient(
                config, temp_output, batch_size
            )
        else:
            import tempfile

            temp_output = tempfile.mkdtemp(prefix="dataset_temp_")
            dataset, metadata = create_dataset_memory_efficient(
                config, temp_output, batch_size
            )
    else:
        logger.info("Using standard processing for smaller dataset...")
        dataset, metadata = create_dataset_from_generator(config, batch_size)

    dataset_name = config["name"]

    if output_dir:
        logger.info(f"Saving dataset locally to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)

        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, "w") as f:
            import json

            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(generate_readme(metadata))
        logger.info(f"Generated README at {readme_path}")

        logger.info("Local save complete")

    if upload_to_hub:
        logger.info(f"Uploading dataset to HuggingFace Hub: {dataset_name}")
        try:
            dataset_card = generate_dataset_card(metadata)

            dataset.push_to_hub(
                dataset_name,
                commit_message=f"Dataset created from config on "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )

            try:
                api = HfApi()

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False
                ) as f:
                    f.write(dataset_card)
                    card_path = f.name
                api.upload_file(
                    path_or_fileobj=card_path,
                    path_in_repo="README.md",
                    repo_id=dataset_name,
                    repo_type="dataset",
                )
                os.unlink(card_path)
                logger.info("Uploaded dataset card to Hub")

            except Exception as card_error:
                logger.warning(f"Could not upload dataset card: {card_error}")

            logger.info("Upload to HuggingFace Hub complete")
        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace Hub: {str(e)}")
            raise

    logger.info("Final dataset created:")
    logger.info(f"  Name: {metadata['name']}")
    logger.info(f"  Description: {metadata['description']}")
    logger.info(f"  Size: {len(dataset):,} entries")
    logger.info(f"  Source datasets: {metadata['source_datasets_count']}")
    logger.info(f"  Columns: {list(dataset.column_names)}")
    logger.info(f"  Creation date: {metadata['creation_date']}")

    logger.info("\nDataset composition:")
    for source in metadata["dataset_sources"]:
        logger.info(
            f"  - {source['name']} ({source['subset']}/{source['split']}): "
            f"{source['actual_entries']:,} entries ({source['percentage_of_total']}%)"
        )


def generate_readme(metadata: Dict[str, Any]) -> str:
    """Generate a comprehensive README for the dataset.

    Args:
        metadata: Dataset metadata dictionary

    Returns:
        README content as markdown string
    """
    readme = f"""# {metadata["name"]}

## Description

{metadata["description"]}

## Dataset Information

- **Total Entries**: {metadata["total_entries"]:,}
- **Source Datasets**: {metadata["source_datasets_count"]}
- **Creation Date**: {metadata["creation_date"]}

{metadata["composition_summary"]}

## Dataset Composition

| Source Dataset | Subset | Split | Requested Entries | Actual Entries | Percentage |
|----------------|---------|-------|------------------|----------------|------------|
"""

    for source in metadata["dataset_sources"]:
        subset_str = source["subset"] if source["subset"] else "N/A"
        readme += f"| {source['name']} | {subset_str} | {source['split']} | {source['requested_entries']:,} | {source['actual_entries']:,} | {source['percentage_of_total']:.2f}% |\n"

    readme += f"""

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{metadata["name"]}")

# Access examples
print(dataset['train'][0])
```

## Source Attribution

This dataset is composed of data from the following sources:

"""

    # Group by unique dataset names
    unique_datasets = {}
    for source in metadata["dataset_sources"]:
        if source["name"] not in unique_datasets:
            unique_datasets[source["name"]] = []
        unique_datasets[source["name"]].append(source)

    for dataset_name, sources in unique_datasets.items():
        readme += f"\n### {dataset_name}\n\n"
        for source in sources:
            subset_str = f" (subset: {source['subset']})" if source["subset"] else ""
            readme += f"- Split: `{source['split']}`{subset_str} - {source['actual_entries']:,} entries\n"

    readme += f"""

---

*This dataset was automatically generated on {metadata["creation_date"]} using a streaming data preparation pipeline.*
"""

    return readme


def generate_dataset_card(metadata: Dict[str, Any]) -> str:
    """Generate a HuggingFace dataset card.

    Args:
        metadata: Dataset metadata dictionary

    Returns:
        Dataset card content as markdown string
    """
    card = f"""---
tags:
- synthetic
- conversation
- instruction-following
size_categories:
- {get_size_category(metadata["total_entries"])}
language:
- en
---

{generate_readme(metadata)}
"""

    return card


def get_size_category(total_entries: int) -> str:
    """Get the appropriate size category for HuggingFace dataset card.

    Args:
        total_entries: Total number of entries in the dataset

    Returns:
        Size category string
    """
    if total_entries < 1000:
        return "n<1K"
    elif total_entries < 10000:
        return "1K<n<10K"
    elif total_entries < 100000:
        return "10K<n<100K"
    elif total_entries < 1000000:
        return "100K<n<1M"
    else:
        return "n>1M"


def main():
    """Main function to process datasets according to configuration."""
    parser = argparse.ArgumentParser(
        description="Create dataset from configuration file using streaming"
    )
    parser.add_argument(
        "--config-path", type=str, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Local directory to save the dataset"
    )
    parser.add_argument(
        "--upload-to-hub", action="store_true", help="Upload dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for processing (default: 5000, increased for efficiency)",
    )
    parser.add_argument(
        "--disable-parquet-optimization",
        action="store_true",
        help="Disable memory-efficient parquet processing for large datasets",
    )
    parser.add_argument(
        "--memory-monitor",
        action="store_true",
        help="Enable memory usage monitoring and logging",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=args.log_level)

    if args.memory_monitor:

        def log_memory_usage():
            while True:
                memory = psutil.virtual_memory()
                logger.info(
                    f"Memory: {memory.percent:.1f}% used "
                    f"({memory.available / (1024**3):.1f}GB available)"
                )
                time.sleep(30)  # Log every 30 seconds

        memory_thread = threading.Thread(target=log_memory_usage, daemon=True)
        memory_thread.start()
        logger.info("Memory monitoring enabled")

    try:
        config = load_config(args.config_path)

        total_requested = sum(d["entries"] for d in config["datasets"])
        logger.info(f"Total requested entries: {total_requested:,}")

        if total_requested > 100000:
            logger.warning(
                f"Large dataset detected ({total_requested:,} entries). "
                f"This may take significant time and disk space."
            )

        save_dataset_streaming(
            config,
            args.output_dir,
            args.upload_to_hub,
            args.batch_size,
            use_parquet_optimization=not args.disable_parquet_optimization,
        )

        logger.info("Dataset preparation completed successfully!")

    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
