from datasets import load_dataset

# Dictionary mapping dataset code names to their actual dataset identifiers
DATASETS = {
    "cifar10": "uoft-cs/cifar10",
}


def load_ds(ds_code="cifar10", split="train"):
    # Check if the dataset_code is valid
    if ds_code not in DATASETS:
        raise ValueError(f"Dataset code '{ds_code}' not found. Available options: {list(DATASETS.keys())}")

    # Load the dataset using the dictionary mapping
    dataset = load_dataset(DATASETS[ds_code])

    # Return the requested split (train/test)
    return dataset[split]


if __name__ == "__main__":
    # Example usage: Load CIFAR-10 dataset and get a sample
    dataset_code = "cifar10"  # Can be changed to "mnist", "celeba", etc.
    ds = load_ds(split="train")
