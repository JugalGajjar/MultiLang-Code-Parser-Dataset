"""
Code Parser for Multiple Programming Languages

This module provides functionality to parse source code files in various programming 
languages using Tree-sitter parsers. It processes parquet files containing code data,
generates Abstract Syntax Trees (ASTs), and saves the parsed results back to parquet format.

Supported Languages: C, C++, C#, Go, Java, JavaScript, Python, Ruby, Scala, TypeScript
"""

import logging
from datetime import datetime
from pathlib import Path
import warnings
from typing import Tuple, List, Optional
import sys

import pandas as pd
from tree_sitter import Language, Parser, Node


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
    log_file = f"{log_dir}/code_parser_{timestamp}.log"
    
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


def create_output_directory(output_dir: str = "./parsed_data_parquet") -> None:
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
        "c", "cpp", "c_sharp", "go", "java", "javascript", 
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


def save_parquet_data(df: pd.DataFrame, language: str, output_dir: str = "./parsed_data_parquet") -> None:
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
    
    try:
        split_index1 = len(df) // 4
        split_index2 = len(df) // 2
        split_index3 = 3 * len(df) // 4

        df1 = df.iloc[:split_index1].copy()
        filename = f"{output_dir}/{language}_parsed_1.parquet"
        logger.info(f"Saving {len(df1)} rows to {filename}.")
        df1.to_parquet(filename, index=False)
        del df1  # Free memory

        df2 = df.iloc[split_index1:split_index2].copy()
        filename = f"{output_dir}/{language}_parsed_2.parquet"
        logger.info(f"Saving {len(df2)} rows to {filename}.")
        df2.to_parquet(filename, index=False)
        del df2  # Free memory

        df3 = df.iloc[split_index2:split_index3].copy()
        filename = f"{output_dir}/{language}_parsed_3.parquet"
        logger.info(f"Saving {len(df3)} rows to {filename}.")
        df3.to_parquet(filename, index=False)
        del df3  # Free memory

        df4 = df.iloc[split_index3:].copy()
        filename = f"{output_dir}/{language}_parsed_4.parquet"
        logger.info(f"Saving {len(df4)} rows to {filename}.")
        df4.to_parquet(filename, index=False)
        del df4  # Free memory

        logger.info(f"All data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving file: {filename}. Error: {e}")
        raise


def walk(code: str, node: Node, indent: int = 0, output: Optional[Tuple[List[str], int]] = None):
    """
    Recursively traverse an Abstract Syntax Tree (AST) and generate a text representation.

    Args:
        code: Original source code as bytes for extracting node snippets
        node: Current Tree-sitter node being processed
        indent: Current indentation level for pretty printing. Defaults to 0
        output: List to accumulate output lines. Created automatically if None.

    Returns:
        Tuple[List[str], int]: A tuple containing the output lines and the total count of nodes in the subtree
    """
    if output is None:
        output = []
    current_node_count = 1

    snippet = code[node.start_byte:node.end_byte].replace("\n", "\\n")  # Replace newlines for better formatting
    output.append("  " * indent + f"({node.type}) \"{snippet}\"")
    for child in node.children:
        _, child_subtree_count = walk(code, child, indent + 1, output)  # Recursive call to process child nodes
        current_node_count += child_subtree_count

    return output, current_node_count


def parse_row(code: str, parser: Parser) -> tuple[str, int, int]:
    """
    Parse a single code sample and extract its Abstract Syntax Tree (AST) representation.
    
    Args:
        code: Source code as a string to be parsed
        parser: Tree-sitter parser instance configured for the specific language
    
    Returns:
        tuple[str, int, int]: A tuple containing the AST representation as a string,
                              the total number of AST nodes, and the number of errors found
    """
    try:
        tree = parser.parse(code.encode())
    except Exception as e:
        logger.error(f"Error parsing code: {e}")
        return None, None, None
    
    output, node_count = walk(code, tree.root_node)
    output = "\n".join(output)

    num_errors = output.count("(ERROR)")  # Count the number of error nodes in the output
    
    return output, node_count, num_errors


def parse_code(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Parse a single code sample and extract AST information.

    Args:
        df: DataFrame containing code samples
        language: Programming language identifier
    
    Returns:
        DataFrame with additional columns for parsed code, AST node count, and error count
    """
    logger.info(f"Parsing code for {language.upper()}.")

    with warnings.catch_warnings():  # Suppress warnings from Tree-sitter
        warnings.simplefilter("ignore")
        LANGUAGE = Language("grammars/languages.so", language)
    
    parser = Parser()
    parser.set_language(LANGUAGE)

    df[["lang_specific_parse", "ast_node_count", "num_errors"]] = df["code"].apply(
        lambda x: pd.Series(parse_row(x, parser))
    )
    logger.info("New features added: 'lang_specific_parse', 'ast_node_count', 'num_errors'.")
    logger.info(f"Parsing completed for {language.upper()}.")

    return df


def process_single_language(language: str, data_dir: str = "./filtered_data_parquet",
                  output_dir: str = "./parsed_data_parquet") -> bool:
    """
    Process a single programming language by loading, parsing, and saving the data.
    
    Args:
        language: Programming language identifier
        data_dir: Directory containing input parquet files
        output_dir: Directory where processed files will be saved
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Starting parsing pipeline for {language.upper()}")
        
        # Load the data
        df = load_parquet_data(language, data_dir)
        
        # Parse the data
        parsed_df = parse_code(df, language)
        del df  # Free memory after parsing
        
        # Save the filtered data
        save_parquet_data(parsed_df, language, output_dir)
        
        logger.info(f"Successfully completed processing for {language}")
        del parsed_df  # Free memory after saving
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {language}: {e}")
        return False


def main():
    """
    Main function to orchestrate the parsing of multiple programming languages.
    """
    sys.setrecursionlimit(10000)  # Increase recursion limit for deep ASTs
    logger.info("Recursion limit set to 10000 for deep AST parsing.")

    # List of supported programming languages
    languages = [
        "c", "cpp", "c_sharp", "go", "java", "javascript", 
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