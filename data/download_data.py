from datasets import load_dataset

def download_and_preprocess():
    # Load the dataset
    dataset = load_dataset("opus100", "en-hr")

    # Split the dataset
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Save datasets to disk (optional, for large datasets)
    train_dataset.save_to_disk("data/train_dataset")
    test_dataset.save_to_disk("data/test_dataset")

    print("Datasets downloaded and preprocessed.")

if __name__ == "__main__":
    download_and_preprocess()
