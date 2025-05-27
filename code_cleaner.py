"""
Code Data Cleaner

This module processes Parquet files containing code data across multiple programming 
languages, removing unwanted content such as repository metadata, issue tracking 
information, and other artifacts that may contaminate the dataset.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import re


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
    log_file = f"{log_dir}/code_cleaner_{timestamp}.log"
    
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
    logger.info(f"Loading data from {filename}.")
    
    try:
        df = pd.read_parquet(filename)
        logger.info(f"Successfully loaded {len(df)} rows from {filename}.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filename}.")
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
        logger.error("Cannot save empty DataFrame.")
        raise ValueError("Cannot save empty DataFrame")
    
    filename = f"{output_dir}/{language}.parquet"
    logger.info(f"Saving {len(df)} rows to {filename}.")
    
    try:
        df.to_parquet(filename, index=False)
        logger.info(f"Data saved successfully to {filename}.")
    except Exception as e:
        logger.error(f"Error saving file: {filename}. Error: {e}")
        raise


def count_substrings_in_column(df: pd.DataFrame, column_name: str, substrings: list[str]) -> int:
    """
    Counts the number of rows in a DataFrame column that contain any of the specified substrings.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        column_name (str): The name of the column to search within.
        substrings (list[str]): A list of substrings to search for.

    Returns:
        int: The count of rows containing any of the substrings.
    """
    if column_name not in df.columns:
        logger.error(f"Error: Column '{column_name}' not found in the DataFrame.")
        return 0
    
    if not substrings:
        logger.warning("Empty substrings list provided.")
        return 0
    
    # Use re.escape to handle special regex characters in substrings
    pattern = "|".join(map(re.escape, substrings))

    # Count rows containing any pattern, handling NaN values
    count = df[column_name].astype(str).str.contains(pattern, na=False, regex=True).sum()

    return int(count)


def drop_rows_with_substrings(df: pd.DataFrame, column_name: str, substrings: list[str]) -> pd.DataFrame:
    """
    Drops rows from the DataFrame if the specified column contains any
    of the given substrings. Returns a new DataFrame.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        column_name (str): The name of the column to search within for dropping rows.
        substrings (list[str]): A list of substrings. If any are found in a row's
                                 column_name cell, that row will be dropped.

    Returns:
        pd.DataFrame: A new DataFrame with rows dropped.
    """
    if column_name not in df.columns:
        logger.error(f"Error: Column '{column_name}' not found in the DataFrame. No rows dropped.")
        return df.copy()
    
    if not substrings:
        logger.warning("Empty substrings list provided. Returning original DataFrame.")
        return df.copy()

    initial_count = len(df)

    # Create a regex pattern to search for any of the substrings
    pattern = "|".join(map(re.escape, substrings))

    # Create a boolean mask: True for rows that contain any of the substrings
    # astype(str) handles potential non-string types and NaN values
    mask_to_drop = df[column_name].astype(str).str.contains(pattern, na=False, regex=True)

    # Return new DataFrame with unwanted rows removed
    filtered_df = df[~mask_to_drop].copy()
    dropped_count = initial_count - len(filtered_df)

    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows from DataFrame where column '{column_name}' contained any of {substrings}.")
    else:
        logger.info(f"No rows found in column '{column_name}' containing any of {substrings} to drop.")

    return filtered_df


def remove_lines_with_substrings(df: pd.DataFrame, column_name: str, substrings: list[str]) -> pd.DataFrame:
    """
    Modifies the specified DataFrame column by removing lines containing substrings.
    For each cell, it splits the string by newline, removes lines containing any
    of the specified substrings, and then rejoins the remaining lines.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        column_name (str): The name of the column to modify.
        substrings (list[str]): A list of substrings to search for and remove lines.

    Returns:
        pd.DataFrame: A new DataFrame with lines removed.
    """
    if column_name not in df.columns:
        logger.error(f"Error: Column '{column_name}' not found in the DataFrame. No modification performed.")
        return df.copy()
    
    if not substrings:
        logger.warning("Empty substrings list provided. Returning original DataFrame.")
        return df.copy()

    # Create a regex pattern to search for any of the substrings
    pattern = "|".join(map(re.escape, substrings))
    compiled_pattern = re.compile(pattern)

    # Define a helper function to process each cell
    def process_cell(cell_value):
        # Handle NaN or None values gracefully
        if pd.isna(cell_value) or cell_value is None:
            return cell_value

        text = str(cell_value)  # Ensure the cell content is treated as a string
        lines = text.split("\n")

        updated_lines = []
        for line in lines:
            # If the line does NOT contain the pattern, add it to updated_lines
            if not compiled_pattern.search(line):
                updated_lines.append(line)

        return "\n".join(updated_lines)

    # Create a copy and apply the cleaning function
    result_df = df.copy()
    result_df[column_name] = df[column_name].apply(process_cell)
    logger.info(f"Column '{column_name}' has been modified to remove lines containing {substrings}.")

    return result_df


def replace_substring(df: pd.DataFrame, column_name: str, old_substring: str, new_substring: str = "") -> pd.DataFrame:
    """
    Replaces all occurrences of a specified substring with another substring
    (defaulting to an empty string) in a DataFrame column.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        column_name (str): The name of the column to modify.
        old_substring (str): The substring to be replaced.
        new_substring (str, optional): The substring to replace with. Defaults to "".
    
    Returns:
        pd.DataFrame: A new DataFrame with replacements made.
    """
    if column_name not in df.columns:
        logger.error(f"Error: Column '{column_name}' not found in the DataFrame. No replacement performed.")
        return df.copy()
    
    if not old_substring:
        logger.warning("Empty old_substring provided. Returning original DataFrame.")
        return df.copy()

    # Create a copy and perform replacement
    result_df = df.copy()
    
    # Use literal string replacement to avoid regex interpretation
    result_df[column_name] = df[column_name].astype(str).str.replace(old_substring, new_substring, regex=False)
    logger.info(f"Replaced all occurrences of '{old_substring}' with '{new_substring}' in column '{column_name}'.")

    return result_df


def filter_data(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Filter DataFrame based on specific criteria for cleaning code data.
    
    Args:
        df: DataFrame to filter
        language: Programming language being processed
    
    Returns:
        Pandas DataFrame containing the filtered data
    """
    logger.info(f"Filtering data for {language}.")

    # Define patterns to remove/clean
    remove_substrings = ["<reponame>", "<gh_stars>", "<filename>"]
    end_of_text_substrings = ["<|endoftext|>"]
    drop_substrings = [
        "<issue_start>", "<issue_comment>", "<issue_closed>", 
        "<jupyter_start>", "<jupyter_text>", "<jupyter_code>", "<jupyter_output>",
        "<commit_before>", "<commit_msg>", "<commit_after>"
    ]
    
    # Phase 1: Drop rows with unwanted content types
    drop_rows_count = count_substrings_in_column(df, "code", drop_substrings)
    logger.info(f"Number of rows in 'code' column containing any of {drop_substrings}: {drop_rows_count}")
    
    if drop_rows_count > 0:
        logger.info(f"Number of rows in 'code' column before dropping: {len(df)}")
        filtered_df = drop_rows_with_substrings(df, "code", drop_substrings)
        logger.info(f"Number of rows in 'code' column after dropping: {len(filtered_df)}")
    else:
        logger.info(f"No rows in 'code' column contain any of {drop_substrings}. No rows dropped.")
        filtered_df = df.copy()
    
    # Phase 2: Clean metadata lines from remaining content
    to_be_cleaned_rows_count = count_substrings_in_column(filtered_df, "code", remove_substrings)
    logger.info(f"Number of rows in 'code' column containing any of {remove_substrings}: {to_be_cleaned_rows_count}")
    
    if to_be_cleaned_rows_count > 0:
        logger.info(f"Number of rows in 'code' column before cleaning phase 1: {len(filtered_df)}")
        filtered_df = remove_lines_with_substrings(filtered_df, "code", remove_substrings)
        logger.info(f"Number of rows in 'code' column after cleaning phase 1: {len(filtered_df)}")
    else:
        logger.info(f"No rows in 'code' column contain any of {remove_substrings}. No lines removed.")
    
    # Phase 3: Remove end-of-text markers
    eot_rows_count = count_substrings_in_column(filtered_df, "code", end_of_text_substrings)
    logger.info(f"Number of rows in 'code' column containing any of {end_of_text_substrings}: {eot_rows_count}")
    
    if eot_rows_count > 0:
        logger.info(f"Number of rows in 'code' column before cleaning phase 2: {len(filtered_df)}")
        filtered_df = replace_substring(filtered_df, "code", end_of_text_substrings[0], "")
        logger.info(f"Number of rows in 'code' column after cleaning phase 2: {len(filtered_df)}")
    else:
        logger.info(f"No rows in 'code' column contain any of {end_of_text_substrings}. No modifications done.")

    # Log final statistics
    initial_count = len(df)
    final_count = len(filtered_df)
    removed_count = initial_count - final_count
    removal_percentage = (removed_count / initial_count) * 100 if initial_count > 0 else 0
    
    logger.info(f"Filtering completed for {language}: {initial_count} -> {final_count} rows "
                f"({removed_count} removed, {removal_percentage:.1f}%)")

    return filtered_df


def process_single_language(language: str, data_dir: str = "./filtered_data_parquet", 
                          output_dir: str = "./filtered_data_parquet") -> bool:
    """
    Process a single programming language dataset through the complete pipeline.
    
    Args:
        language: Programming language to process
        data_dir: Directory containing input parquet files
        output_dir: Directory where processed files will be saved
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Starting processing pipeline for {language}")
        
        # Load the data
        df = load_parquet_data(language, data_dir)
        
        # Filter the data
        filtered_df = filter_data(df, language)
        
        # Save the filtered data
        save_parquet_data(filtered_df, language, output_dir)
        
        logger.info(f"Successfully completed processing for {language}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {language}: {e}")
        return False


def main():
    """
    Main function to clean code data for all supported languages.
    
    Args:
        None
    """
    # List of supported programming languages
    languages = [
        "c", "cpp", "c-sharp", "go", "java", "javascript", 
        "python", "ruby", "scala", "typescript"
    ]
    
    # Ensure output directory exists
    create_output_directory()
    
    # Track processing results
    results = {}
    
    logger.info(f"Starting batch processing for {len(languages)} languages")
    
    for language in languages:
        try:
            # Process each language
            success = process_single_language(language)
            results[language] = success
            
        except Exception as e:
            logger.error(f"An error occurred while processing {language}: {e}")
            results[language] = False
            continue
    
    # Generate final summary
    successful_languages = [lang for lang, success in results.items() if success]
    failed_languages = [lang for lang, success in results.items() if not success]
    
    logger.info("Data processing completed.")
    logger.info(f"Successfully processed: {len(successful_languages)}/{len(languages)} languages")
    
    if successful_languages:
        logger.info(f"Successful: {', '.join(successful_languages)}")
    
    if failed_languages:
        logger.error(f"Failed: {', '.join(failed_languages)}")



if __name__ == "__main__":
    main()