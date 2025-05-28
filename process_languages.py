"""
Language Data Processor

This script processes JSON files containing code data for multiple programming languages
and converts them to parquet format. It extracts code samples, average line length,
and line count for each language.

Usage:
    python process_languages.py
"""

import json
import logging
import os
from typing import Dict, List, Any
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
    log_file = f"{log_dir}/language_processor_{timestamp}.log"
    
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

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


def setup_output_directory(directory: str = "./filtered_data_parquet") -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        directory: Name of the output directory
        
    Returns:
        Path to the output directory
    """
    if not os.path.exists(directory):
        logger.info(f"Creating output directory: {directory}.")
        os.makedirs(directory)
    return directory


def load_json_data(language: str) -> Dict[str, Any]:
    """
    Load JSON data for a specific language.
    
    Args:
        language: Programming language identifier
        
    Returns:
        Dictionary containing the loaded JSON data
        
    Raises:
        FileNotFoundError: If the JSON file for the language doesn't exist
        json.JSONDecodeError: If the JSON file is invalid
    """
    filename = f"./filtered_data/{language}.json"
    logger.info(f"Loading data from {filename}.")
    
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filename}: {e}")
        raise


def process_language_data(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process language JSON data and extract relevant fields.
    
    Args:
        json_data: Dictionary containing language data
        
    Returns:
        List of dictionaries with extracted data
    """
    processed_data = []
    
    for lang_id, content in json_data.items():
        processed_data.append({
            "language": lang_id.split("_")[0],
            "code": content["code"],
            "avg_line_length": content["avg_line_length"],
            "line_count": content["line_count"]
        })
    
    return processed_data


def convert_to_parquet(
    data: List[Dict[str, Any]],
    language: str,
    output_dir: str = "./filtered_data_parquet"
) -> str:
    """
    Convert processed data to parquet format and save to disk.
    
    Args:
        data: List of dictionaries containing processed data
        language: Programming language identifier
        output_dir: Directory to save the parquet file
        
    Returns:
        Path to the saved parquet file
    """
    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, f"{language}.parquet")
    
    logger.info(f"Converting {language.upper()} data to parquet format.")
    df.to_parquet(output_path)
    logger.info(f"Saved parquet file to {output_path}.")
    
    return output_path


def main():
    """
    Main function to process all languages and convert data to parquet format.
    """
    languages = [
        "c", "cpp", "c-sharp", "go", "java", "javascript", 
        "python", "ruby", "scala", "typescript"
    ]
    
    logger.info(f"Starting processing for {len(languages)} languages.")
    
    output_dir = setup_output_directory()
    processed_count = 0
    failed_count = 0
    
    for language in languages:
        try:
            logger.info(f"Processing {language.upper()} data...")
            # Load JSON data
            json_data = load_json_data(language)
            
            # Process the data
            processed_data = process_language_data(json_data)
            
            # Convert to parquet and save
            output_path = convert_to_parquet(processed_data, language, output_dir)
            
            processed_count += 1
            logger.info(f"Successfully processed {language.upper()} language.")
            
        except Exception as e:
            failed_count += 1
            logger.error(f"Error processing {language.upper()}: {e}")
    
    logger.info(f"Processing complete. Successfully processed {processed_count} of {len(languages)} languages.")
    if failed_count > 0:
        logger.warning(f"Failed to process {failed_count} languages. See log for details.")



if __name__ == "__main__":
    main()