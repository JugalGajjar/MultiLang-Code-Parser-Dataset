"""
Code Data Filtering Script

This script filters and processes programming language datasets for code analysis.
It loads datasets from disk, calculates statistical measures, removes outliers,
generates visualizations, and saves filtered data in JSON format.

Features:
- Statistical analysis of code metrics (avg line length, line count)
- Outlier detection and removal using IQR method
- Data visualization with boxplots and histograms
- Multi-language dataset processing with partitioning options
- Configurable random seed for reproducibility
"""

import os
import json
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk, Dataset


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
    log_file = f"{log_dir}/data_filter_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def save_json(save_path: str, data: Dict) -> bool:
    """
    Save dictionary data to a JSON file.
    
    Args:
        save_path: Path where the JSON file will be saved
        data: Dictionary containing data to be saved
        
    Returns:
        Boolean indicating whether the save operation was successful
        
    Raises:
        Logs any errors encountered during the save operation
    """
    try:
        with open(save_path, "w") as f:
            json.dump(data, f)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")
        return False


def shuffle_data(dataset: Dataset, seed: int = 42) -> Dataset:
    """
    Shuffles a dataset for randomization.

    Args:
        dataset: The dataset to shuffle
        seed: Random seed for reproducibility

    Returns:
        A shuffled dataset with the same content but in random order
    """
    # Initialize random generator with seed for reproducibility
    generator = random.Random(seed)
    
    # Create a list of indices and shuffle them
    shuffled_indices = list(range(len(dataset)))
    generator.shuffle(shuffled_indices)

    # Return dataset with reordered indices
    return dataset.select(shuffled_indices)


def calculate_avg_line_length(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the average line length for a code example.
    
    Args:
        example: Dictionary containing code content
        
    Returns:
        The same dictionary with an added 'avg_line_length' field
    """
    # Split the content into lines and calculate average length
    lines = example["content"].split("\n")
    # Avoid division by zero for empty content
    if len(lines) > 0:
        avg_length = sum(len(line) for line in lines) / len(lines)
    else:
        avg_length = 0
    
    # Add average line length as a new feature
    example["avg_line_length"] = avg_length
    return example


def log_statistics_and_filter(
    lang: str,
    dataset: Dataset,
    plot_save_directory: str = "./graphs"
) -> Dict[str, Dict[str, Any]]:
    """
    Generate statistics, create visualizations, and filter the dataset.
    
    Args:
        lang: Programming language identifier
        dataset: Dataset containing code samples
        plot_save_directory: Directory to save visualization plots
        
    Returns:
        Dictionary containing filtered code data with metrics
    """
    # Create the save directory if it doesn't exist
    Path(plot_save_directory).mkdir(exist_ok=True)
    
    # Convert dataset to pandas DataFrame for easier analysis
    df = pd.DataFrame(dataset)
    
    # Calculate and log statistics for average line length
    avg_line_length_stats = df["avg_line_length"].describe()
    logging.info(f"Statistics for avg_line_length:\n{avg_line_length_stats}")
    
    # Create and save boxplot for average line length
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df["avg_line_length"])
    plt.title(f"Boxplot of Average Line Length for {lang.upper()}")
    plt.xlabel("Average Line Length")
    plot_path = f"{plot_save_directory}/{lang}_box_plot_avg_line_length.png"
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    logging.info(f"Box plot for average line length saved to {plot_path}")
    
    # Filter outliers using IQR method
    Q1 = df["avg_line_length"].quantile(0.25)
    Q3 = df["avg_line_length"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df["avg_line_length"] < (Q1 - 1.5 * IQR)) | 
                  (df["avg_line_length"] > (Q3 + 1.5 * IQR))]
    logging.info(f"Number of outliers identified: {len(outliers)}")
    logging.info("Dropping outliers from the dataset.")
    
    # Remove outliers from the dataset
    df.drop(outliers.index, inplace=True)
    
    # Log updated statistics after outlier removal
    new_avg_line_length_stats = df["avg_line_length"].describe()
    logging.info(f"Statistics for avg_line_length after outlier removal:\n{new_avg_line_length_stats}")
    
    # Add line count feature to the dataset
    df["line_count"] = df["content"].apply(lambda x: len(x.split("\n")))
    logging.info("'line_count' feature added to the dataset.")
    
    # Calculate and log statistics for line count
    line_count_stats = df["line_count"].describe()
    logging.info(f"Statistics for line_count:\n{line_count_stats}")
    
    # Create distribution of code lengths
    count_range_dict = {
        "0-100": 0, "101-200": 0, "201-300": 0, "301-400": 0, "401-500": 0,
        "501-600": 0, "601-700": 0, "701-800": 0, "801-900": 0, "901-1000": 0,
        "1000+": 0
    }
    
    # Count samples in each size range
    for count in df["line_count"]:
        if count <= 100:
            count_range_dict["0-100"] += 1
        elif count <= 200:
            count_range_dict["101-200"] += 1
        elif count <= 300:
            count_range_dict["201-300"] += 1
        elif count <= 400:
            count_range_dict["301-400"] += 1
        elif count <= 500:
            count_range_dict["401-500"] += 1
        elif count <= 600:
            count_range_dict["501-600"] += 1
        elif count <= 700:
            count_range_dict["601-700"] += 1
        elif count <= 800:
            count_range_dict["701-800"] += 1
        elif count <= 900:
            count_range_dict["801-900"] += 1
        elif count <= 1000:
            count_range_dict["901-1000"] += 1
        else:
            count_range_dict["1000+"] += 1
    
    # Plot distribution of code lengths
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(count_range_dict.keys()), y=list(count_range_dict.values()))
    plt.title(f"Distribution of {lang.upper()} Code Lengths")
    plt.xlabel("Length of Code (lines)")
    plt.ylabel("Number of Examples")
    plt.xticks(rotation=45)
    
    # Save the plot
    dist_plot_path = f"{plot_save_directory}/{lang}_distribution_of_code_lengths.png"
    plt.savefig(dist_plot_path, bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    logging.info(f"Distribution of code lengths saved to {dist_plot_path}")
    
    # Filter the dataset by removing unnecessary columns and filtering by thresholds
    columns_to_drop = [col for col in ["max_stars_repo_path", "max_stars_repo_name", 
                                    "max_stars_count", "id"] if col in df.columns]
    df_filtered = df.drop(columns=columns_to_drop)
    
    df_filtered = df_filtered[df_filtered["line_count"] <= np.percentile(df_filtered["line_count"], 90)]
    df_filtered = df_filtered[df_filtered["avg_line_length"] <= np.percentile(df_filtered["avg_line_length"], 80)]
    
    # Log comparison of statistics before and after filtering
    logging.info("Statistics comparison for 'avg_line_length' and 'line_count' before and after filtering:")
    logging.info(f"BEFORE: \n{df[['avg_line_length', 'line_count']].describe()}")
    logging.info(f"AFTER: \n{df_filtered[['avg_line_length', 'line_count']].describe()}")
    
    # Create final data dictionary with unique IDs
    final_data = {}
    for i, (ind, row) in enumerate(df_filtered.iterrows()):
        final_data[f"{lang}_{i}"] = {
            "code": row["content"],
            "avg_line_length": row["avg_line_length"],
            "line_count": row["line_count"]
        }
    
    return final_data


def load_code_data(
    languages: List[str],
    save_directory: str = "./filtered_data",
    seed: int = 42
) -> Dict[str, Optional[bool]]:
    """
    Load, process and filter code data for multiple programming languages.
    
    Args:
        languages: List of programming language identifiers to process
        save_directory: Directory to save the filtered datasets
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping language identifiers to processing success status
    """
    # Create the save directory if it doesn't exist
    Path(save_directory).mkdir(exist_ok=True)
    
    # Log processing information
    logging.info(f"Number of languages currently filtering: {len(languages)}")
    logging.info("Languages included:")
    for lang in languages:
        logging.info(f"  - {lang.upper()}")
    
    # Track success status for each language
    success_status = {}
    
    # Process each language dataset
    for lang in languages:
        try:
            logging.info(f"Working with {lang.upper()} language...")
            
            # Load dataset from disk
            ds = load_from_disk(f"./{lang}_train_dataset")
            logging.info("  - Successfully loaded the code data.")
            
            # Shuffle the dataset for randomization
            ds = shuffle_data(dataset=ds, seed=seed)
            logging.info("  - Data shuffling completed.")
            
            # Calculate average line length for each sample
            ds = ds.map(calculate_avg_line_length)
            logging.info("  - 'avg_line_length' feature added to the dataset.")
            
            ds = ds[:2500000]  # Take first 2.5M samples
            logging.info("  - Data slicing completed.")
            
            # Filter and analyze the dataset
            logging.info("  - Working with data statistics and filtering based on it.")
            filtered_data = log_statistics_and_filter(lang=lang, dataset=ds)
            logging.info("  - Data filtered successfully.")
            
            # Save filtered data to disk
            logging.info("  - Writing filtered data to disk.")
            save_path = f"{save_directory}/{lang}.json"
            save_success = save_json(save_path, filtered_data)
            
            if save_success:
                logging.info(f"  - Data successfully saved at {save_path}.")
                # Mark as successful
                success_status[lang] = True
            else:
                logging.error(f"  - Failed to save data at {save_path}.")
                success_status[lang] = False
            
        except Exception as e:
            logging.error(f"Error processing {lang} dataset: {e}")
            success_status[lang] = False
    
    return success_status


def main():
    """
    Main function to parse arguments and initialize the code filtering process.
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Code Dataset Filter - Processes and filters programming language datasets")
    
    parser.add_argument(
        "-d", "--process-in-parts", 
        action="store_true",
        help="Process data in parts instead of all at once. Useful for large datasets.")
    
    parser.add_argument(
        "-p", "--partition-num", 
        type=int,
        default=0,
        choices=[1, 2, 3, 4, 5],
        help="Partition number (1-5) to process specific languages. Each partition contains 2 languages.")
    
    parser.add_argument(
        "-s", "--random-seed", 
        type=int,
        default=42,
        help="Seed value for randomly shuffling the code data.")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()

    # Define all supported programming languages
    languages = [
        "c", "cpp", "c-sharp", "go", "java", "javascript", 
        "python", "ruby", "scala", "typescript"
    ]

    # Apply partitioning if specified
    selected_languages = languages
    if args.process_in_parts and args.partition_num > 0:
        logging.info(f"Processing partition {args.partition_num} of 5")
        
        # Select languages based on partition number
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
    
    # Process selected language datasets
    success_dict = load_code_data(
        languages=selected_languages,
        save_directory="./filtered_data",
        seed=args.random_seed
    )

    # Report final statistics
    successful_filters = sum(1 for success in success_dict.values() if success)
    logging.info(f"Filter process completed. Successfully filtered {successful_filters}/{len(selected_languages)} datasets.")
    
    # Log any failed languages
    failed_languages = [lang for lang, success in success_dict.items() if not success]
    if failed_languages:
        logging.warning(f"Failed to process these languages: {', '.join(failed_languages)}")


if __name__ == "__main__":
    main()
