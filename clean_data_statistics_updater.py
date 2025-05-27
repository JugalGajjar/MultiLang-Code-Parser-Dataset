"""
Code Metrics Updater

This module processes programming language datasets stored in Parquet format,
calculates code metrics (average line length and line count), and saves the
updated data back to Parquet files along with logging.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


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
    log_file = f"{log_dir}/clean_stats_updater_{timestamp}.log"
    
    # Configure logging with detailed format including function names
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ],
        force=True  # Override any existing configuration
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


def create_output_directory(output_dir: str = "./filtered_data_parquet") -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Directory where processed files will be saved
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {output_dir}")


def load_parquet_data(language: str, data_dir: str = "./filtered_data_parquet") -> pd.DataFrame:
    """
    Load Parquet data for a specific language.
    
    Args:
        language: Programming language identifier
        data_dir: Directory containing input parquet files
        
    Returns:
        Pandas DataFrame containing the loaded Parquet data
        
    Raises:
        FileNotFoundError: If the parquet file for the language doesn't exist
        ValueError: If the language is not supported
    """
    # List of supported languages
    supported_languages = [
        "c", "cpp", "c-sharp", "go", "java", "javascript", 
        "python", "ruby", "scala", "typescript"
    ]
    
    if language not in supported_languages:
        logger.error(f"Unsupported language: {language}. Supported: {supported_languages}")
        raise ValueError(f"Unsupported language: {language}")
    
    filename = f"{data_dir}/{language}.parquet"
    logger.info(f"Loading data from {filename}")
    
    try:
        df = pd.read_parquet(filename)
        logger.info(f"Successfully loaded {len(df)} rows from {filename}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {filename}. Error: {e}")
        raise


def save_parquet_data(df: pd.DataFrame, language: str, output_dir: str = "./filtered_data_parquet") -> None:
    """
    Save DataFrame to Parquet format with compression.
    
    Args:
        df: DataFrame to save
        language: Programming language identifier for filename
        output_dir: Directory where processed files will be saved
        
    Raises:
        ValueError: If DataFrame is empty
    """
    if df.empty:
        logger.error("Cannot save empty DataFrame")
        raise ValueError("Cannot save empty DataFrame")
    
    filename = f"{output_dir}/{language}.parquet"
    logger.info(f"Saving {len(df)} rows to {filename}")
    
    try:
        df.to_parquet(filename, index=False)
        logger.info(f"Data saved successfully to {filename}")
    except Exception as e:
        logger.error(f"Error saving file: {filename}. Error: {e}")
        raise


def calculate_code_metrics(code: str) -> tuple[float, int]:
    """
    Calculate average line length and line count for a code string.

    Args:
        code: The code string to analyze

    Returns:
        Tuple containing (avg_line_length, line_count)
    """
    merge_flag = False  # Flag to merge lines ending with backslash
    content = ""
    lines = code.strip().split("\n")
    for i, line in enumerate(lines):
        if line == "":
            lines.pop(i) # Remove empty lines
        if line.endswith("\\"):
            content += line[:-1]
            merge_flag = True
        if merge_flag:
            content += line
            lines[i] = content
            lines.pop(i-1)  # Remove the previous line
            content = ""
            merge_flag = False
    line_count = len(lines)

    if line_count == 0:
        return 0.0, 0

    # Calculate total character count across all lines
    total_length = sum(len(line) for line in lines)
    avg_line_length = round(total_length / line_count, 2)
    
    return avg_line_length, line_count


def update_code_metrics_and_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update DataFrame with recalculated code metrics and log statistics.

    Args:
        df: Input DataFrame with 'code' column

    Returns:
        DataFrame with updated 'avg_line_length' and 'line_count' columns
    """
    # Apply the calculation function to update the columns
    df[["avg_line_length", "line_count"]] = df["code"].apply(
        lambda x: pd.Series(calculate_code_metrics(x))
    )

    # Calculate statistics for logging
    total_rows = len(df)
    avg_line_count = df['line_count'].mean()
    avg_avg_line_length = df['avg_line_length'].mean()

    logger.info(f"Total number of rows in the dataset: {total_rows}")
    logger.info(f"Average line count (mean over 'line_count'): {avg_line_count:.2f}")
    logger.info(f"Average average line length (mean over 'avg_line_length'): {avg_avg_line_length:.2f}")

    return df


def process_language_data(language: str, data_dir: str = "./filtered_data_parquet") -> None:
    """
    Process the data for a specific programming language.
    
    Args:
        language: Programming language identifier
        data_dir: Directory containing input parquet files
    """
    # Load the data
    df = load_parquet_data(language, data_dir)
    
    # Update code metrics
    df = update_code_metrics_and_log(df)
    
    # Save the updated DataFrame
    save_parquet_data(df, language, data_dir)
    logger.info(f"Processing completed for {language.upper()} language.")


def main():
    """
    Main function to process data for all supported languages.
    """
    # List of supported languages
    supported_languages = [
        "c", "cpp", "c-sharp", "go", "java", "javascript", 
        "python", "ruby", "scala", "typescript"
    ]
    
    # Create output directory
    create_output_directory()
    
    # Process data for each language
    for language in supported_languages:
        try:
            process_language_data(language)
        except Exception as e:
            logger.error(f"Error processing {language}: {e}")

    logger.info("All languages processed successfully")


if __name__ == "__main__":
    main()