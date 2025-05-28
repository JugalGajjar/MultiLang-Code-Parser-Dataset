"""
StarCoder Dataset Downloader

This script downloads and saves programming language datasets from the bigcode/starcoderdata
collection on Hugging Face for multiple programming languages.

The script uses logging to track the download process and saves logs to a timestamped
file in a 'logs' directory.

The script can download data in parts or as a whole, useful for managing large downloads
that may require significant storage (>= 1TB for the complete dataset).
"""

import os
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

from dotenv import load_dotenv
from datasets import load_dataset, DownloadMode, Dataset
from huggingface_hub import login


def setup_logging(log_dir: str = "./logs") -> None:
    """
    Set up logging configuration with timestamps in filenames.
    
    Args:
        log_dir: Directory where log files will be stored
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create a timestamp for the log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/starcoder_download_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def download_starcoderdata(
    save_directory: str,
    language_dir: str,
    split: str = "train",
    download_mode: DownloadMode = DownloadMode.REUSE_DATASET_IF_EXISTS
) -> Optional[Dataset]:
    """
    Download a specific language dataset from bigcode/starcoderdata.
    
    Args:
        save_directory: Directory to save the downloaded dataset
        language_dir: Programming language directory identifier in the dataset
        split: Dataset split to download (e.g., "train", "test")
        download_mode: Mode to use for downloading the dataset
        
    Returns:
        The downloaded dataset object or None if download failed
    """
    try:
        logging.info(f"Downloading dataset for {language_dir.upper()}...")
        
        # Load the dataset from Hugging Face
        ds = load_dataset(
            "bigcode/starcoderdata",
            data_dir=language_dir,
            split=split,
            cache_dir=save_directory,
            download_mode=download_mode,
        )

        logging.info(f"  - {language_dir.upper()} dataset downloaded successfully")
        
        # Save the dataset to disk
        output_path = f"{save_directory}/{language_dir}_{split}_dataset"
        ds.save_to_disk(output_path)
        logging.info(f"  - {language_dir.upper()} dataset saved to '{output_path}'")

        return ds

    except Exception as e:
        logging.error(f"Error downloading dataset 'bigcode/starcoderdata' ({language_dir}, {split}): {e}")
        return None


def download_all_languages(
    languages: List[str],
    save_directory: str = "./",
    split: str = "train"
) -> Dict[str, Optional[Dataset]]:
    """
    Download datasets for all specified programming languages.
    
    Args:
        languages: List of programming language identifiers to download
        save_directory: Directory to save all downloaded datasets
        split: Dataset split to download
        
    Returns:
        Dictionary mapping language identifiers to their dataset objects
    """
    # Create the save directory if it doesn't exist
    Path(save_directory).mkdir(exist_ok=True)
    
    logging.info(f"Number of languages included in the dataset: {len(languages)}")
    logging.info("Languages included:")
    for lang in languages:
        logging.info(f"- {lang.upper()}")
    
    # Download each language dataset
    data_dict = {}
    for lang in languages:
        ds = download_starcoderdata(
            save_directory=save_directory,
            language_dir=lang,
            split=split
        )
        data_dict[lang] = ds
    
    return data_dict


def main():
    """
    Main function to initialize the script and download all datasets.
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="StarCoder Dataset Downloader - Downloads programming language datasets from Hugging Face")
    parser.add_argument(
        "-d", "--download-in-parts", 
        action="store_true",
        help="Download data in parts instead of all at once. The complete dataset requires >= 1TB storage.")
    parser.add_argument(
        "-p", "--partition-num", 
        type=int,
        default=0,
        choices=[1, 2, 3, 4, 5],
        help="Partition number (1-5) to download specific languages. Each partition contains 2 languages.")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Hugging Face token from environment variables
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.error("HF_TOKEN environment variable not found. Please set it in your .env file.")
        return
    
    # Login to Hugging Face
    logging.info("Logging in to Hugging Face Hub...")
    login(hf_token)
    
    # Define the programming languages to download
    languages = [
        "c", "cpp", "c-sharp", "go", "java", "javascript", 
        "python", "ruby", "scala", "typescript"
    ]

    # Apply partitioning if specified
    selected_languages = languages
    if args.download_in_parts and args.partition_num > 0:
        logging.info(f"Downloading partition {args.partition_num} of 5")
        
        if args.partition_num == 1:
            selected_languages = languages[:2]
        elif args.partition_num == 2:
            selected_languages = languages[2:4]
        elif args.partition_num == 3:
            selected_languages = languages[4:6]
        elif args.partition_num == 4:
            selected_languages = languages[6:8]
        elif args.partition_num == 5:
            selected_languages = languages[8:]
        else:
            logging.error(f"Invalid partition number {args.partition_num}. It should be a number between 1-5 (inclusive).")
            return
    
    # Download selected language datasets
    data_dict = download_all_languages(
        selected_languages
    )
    
    # Check if all downloads were successful
    successful_downloads = sum(1 for ds in data_dict.values() if ds is not None)
    logging.info(f"Download process completed. Successfully downloaded {successful_downloads}/{len(selected_languages)} datasets.")



if __name__ == "__main__":
    main()