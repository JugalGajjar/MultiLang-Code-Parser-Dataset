"""
Tree-Sitter to Universal Schema Conversion Script

This module provides utilities to load compiled Tree-sitter language grammars,
parse source code from parquet datasets into a compact universal AST schema,
categorize AST nodes, and produce a cross-language mapping of declarations.

Features:
- Load multiple compiled Tree-sitter language grammars.
- Test loaded parsers with small sample snippets.
- Traverse and serialize Tree-sitter ASTs into a filtered node list.
- Categorize nodes into declarations/statements/expressions.
- Extract simple identifier names from node text for many languages.
- Produce a minimal cross-language mapping (functions, classes, imports).
- Convert rows from parquet files to a universal JSON schema and write back.
"""

import pandas as pd
import json
import os
import logging
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from tree_sitter import Parser, Node, Language

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
    log_file = f"{log_dir}/parse_converter{timestamp}.log"
    
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

logger.info("Loading compiled grammars from grammars/languages.so...")
LANGUAGE_OBJECTS = {}

languages_to_load = ['python', 'c', 'cpp', 'java', 'javascript', 'typescript', 'go', 'ruby', 'c_sharp', 'scala']
successful_loads = []

for lang in languages_to_load:
    try:
        lang_obj = Language('grammars/languages.so', lang)
        LANGUAGE_OBJECTS[lang] = lang_obj
        successful_loads.append(lang)
        logger.info("Loaded Tree-sitter grammar: %s", lang)
    except Exception as e:
        logger.warning("Failed to load grammar for %s: %s", lang, e)

SUPPORTED_LANGUAGES = {lang: lang for lang in successful_loads}
logger.info("Supported languages: %s", list(SUPPORTED_LANGUAGES.keys()))


def get_parser(language: str) -> Parser:
    """
    Return a configured Tree-sitter Parser for the requested language.

    Args:
        language: Language name (e.g., "python", "c_sharp", "typescript"). 
                  Some aliases like "csharp" or "c-sharp" are normalized.

    Returns:
        A tree_sitter.Parser instance set to the requested language.

    Raises:
        ValueError: If the requested language is not in SUPPORTED_LANGUAGES.
    """
    # Handle language name variations
    language_mapping = {
        'csharp': 'c_sharp',
        'c-sharp': 'c_sharp',
    }
    normalized_lang = language_mapping.get(language, language)
    
    if normalized_lang not in SUPPORTED_LANGUAGES:
        logger.error("Unsupported language requested: %s -> %s", language, normalized_lang)
        raise ValueError(f"Unsupported language: {language} -> {normalized_lang}")
    
    parser = Parser()
    parser.set_language(LANGUAGE_OBJECTS[normalized_lang])
    return parser


def test_all_parsers():
    """
    Test all loaded Tree-sitter parsers with small sample snippets.

    Args:
        None

    Returns:
        Tuple[List[str], List[str]]: (working_parsers, failed_parsers).
    """
    print("\n=== Testing ALL Language Parsers ===")
    test_cases = {
        'python': "print('hello world')",
        'c': "int main() { return 0; }",
        'cpp': "#include <iostream>\nint main() { return 0; }",
        'java': "class Test { public static void main(String[] args) {} }",
        'javascript': "console.log('hello');",
        'typescript': "const message: string = 'hello';",
        'go': "package main\nfunc main() { println('hello') }",
        'ruby': "puts 'hello'",
        'c_sharp': "using System;\nclass Program { static void Main() { Console.WriteLine('hello'); } }",
        'scala': "object Hello { def main(args: Array[String]) = println('hello') }"
    }
    
    working_parsers = []
    failed_parsers = []
    
    for lang in SUPPORTED_LANGUAGES.keys():
        if lang in test_cases:
            try:
                parser = get_parser(lang)
                test_code = test_cases[lang]
                tree = parser.parse(bytes(test_code, 'utf8'))
                working_parsers.append(lang)
                logger.info("Parser OK: %s - root:%s children:%d", lang, tree.root_node.type, len(tree.root_node.children))
            except Exception as e:
                failed_parsers.append(lang)
                logger.error("Parser failed for %s: %s", lang, e)
    
    logger.info("Parser test summary: %d working, %d failed", len(working_parsers), len(failed_parsers))
    return working_parsers, failed_parsers

# Test all parsers
working_parsers, failed_parsers = test_all_parsers()


def hash_string(text: str) -> str:
    """
    Compute SHA-256 hex digest of the input text.

    Args:
        text: Input string to hash.

    Returns:
        Hexadecimal SHA-256 hash as a string.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def calculate_line_offsets(text: str) -> List[int]:
    """
    Compute byte offsets of line starts in a text.

    Args:
        text: Input string.

    Returns:
        List of integer offsets for the start index of each line.
    """
    offsets = [0]
    for i, char in enumerate(text):
        if char == '\n':
            offsets.append(i + 1)
    return offsets


def extract_ast_structure(tree, source_code: str, language: str) -> Dict:
    """
    Traverse a Tree-sitter parse tree and serialize it into a filtered node list.

    Args:
        tree: Tree-sitter parse tree returned by parser.parse(...).
        source_code: Original source code string corresponding to the tree.
        language: Normalized language name used for language-specific rules.

    Returns:
        Dict with keys 'root' (root node type) and 'nodes' (list of serialized nodes).
    """
    nodes = []
    node_id_counter = 0
    
    # Define tokens to skip (pure punctuation/delimiters)
    SKIP_TOKENS = {
        '(', ')', '{', '}', '[', ']', ',', ';', ':', 
        '.', '->', '::', '...', '|', '&',
        # Common keyword-only nodes that are redundant
        'def', 'class', 'function', 'var', 'let', 'const',
        'public', 'private', 'protected', 'static',
        'if', 'else', 'for', 'while', 'return',
        'import', 'from', 'as', 'export', 'package',
        'try', 'catch', 'finally', 'throw', 'raise',
        'new', 'delete', 'sizeof', 'typeof',
        'async', 'await', 'yield',
        # Add language-specific delimiters
        'then', 'do', 'end', 'begin'
    }
    
    # Define node types that should be skipped entirely
    SKIP_NODE_TYPES = {
        # Pure punctuation/delimiter types
        '(', ')', '{', '}', '[', ']', ',', ';', ':', '.',
        # Standalone keyword types that don't add semantic value
        'def', 'class', 'return', 'if', 'else', 'elif',
        'for', 'while', 'import', 'from', 'as',
        'public', 'private', 'protected', 'static',
        'const', 'var', 'let', 'function',
        'try', 'catch', 'finally', 'throw',
        'async', 'await', 'new', 'delete',
        # Common tree-sitter delimiter types
        'comment',  # Keep this if you want comments, remove if not
        'string_content', 'string_start', 'string_end', 'expression_statement',
        'block', 'module', 'dotted_name', 'program', 'block_comment', 'class_body',
        '\"', 'string_fragment', 'translation_unit', 'compound_statement', 'field_declaration_list',
        'compilation_unit', 'declaration_list', 'string_literal_content', 'source_file',
        'interpreted_string_literal_content', 'statement_block', 'type_annotation', '/*', 'template_body'
    }
    
    def should_skip_node(node: Node) -> bool:
        """Determine if a node should be skipped based on granularity rules"""
        # Try to obtain text, fall back to empty string
        try:
            node_text = source_code[node.start_byte:node.end_byte].strip()
        except Exception:
            node_text = ""
        
        node_type = node.type
        
        # Skip if it's a pure punctuation/delimiter token
        if node_text in SKIP_TOKENS:
            return True
        
        # Skip if the node type itself is just punctuation or in skip list
        if node_type in SKIP_NODE_TYPES:
            return True
        
        # Skip large triple-quoted strings (python)
        if node_type == "string" and (
            node_text.startswith('"""') or node_text.startswith("'''")
        ):
            return True

        # Skip redundant variable_declarator if identical to parent text
        if (
            node_type == "variable_declarator" 
            and node.parent is not None 
            and node.parent.type in ("variable_declaration", "lexical_declaration")
        ):
            try:
                parent_text = source_code[node.parent.start_byte:node.parent.end_byte]
                if node_text and node_text in parent_text:
                    return True
            except Exception:
                pass
        
        # Skip redundant type_spec nodes
        if (
            node_type == "type_spec" 
            and node.parent is not None 
            and node.parent.type == "type_declaration"
        ):
            try:
                parent_text = source_code[node.parent.start_byte:node.parent.end_byte]
                if node_text and node_text in parent_text:
                    return True
            except Exception:
                pass
        
        return False
    
    def traverse_node(node: Node, parent_id: Optional[int]) -> Optional[int]:
        nonlocal node_id_counter
        
        # Check if we should skip this node
        if should_skip_node(node):
            # Don't create a node, but still traverse children
            # and connect them to the current parent
            for child in node.children:
                traverse_node(child, parent_id)
            return None
        
        current_id = node_id_counter
        node_id_counter += 1
        
        # Extract node text from source code defensively
        try:
            node_text = source_code[node.start_byte:node.end_byte]
        except Exception:
            node_text = ""
        
        # Build node data
        node_data = {
            "id": current_id,
            "type": node.type,
            "text": node_text,
            "parent": parent_id,
            "children": [],
            "start_point": {"row": node.start_point[0], "column": node.start_point[1]},
            "end_point": {"row": node.end_point[0], "column": node.end_point[1]}
        }
        
        nodes.append(node_data)
        
        # Recursively traverse children
        child_ids = []
        for child in node.children:
            child_id = traverse_node(child, current_id)
            if child_id is not None:  # Only add non-skipped children
                child_ids.append(child_id)
        
        nodes[current_id]["children"] = child_ids
        return current_id
    
    # Start traversal from root
    root_id = traverse_node(tree.root_node, None)
    
    return {
        "root": tree.root_node.type,
        "nodes": nodes
    }


def categorize_nodes_by_keywords(ast_structure: Dict, language: str) -> Dict:
    """
    Classify AST nodes into declaration/statement/expression categories by node type.

    Args:
        ast_structure: Serialized AST structure produced by extract_ast_structure.
        language: Normalized language name (currently unused in heuristics).

    Returns:
        Dictionary of categories (declarations, statements, expressions) each
        containing lists of node IDs belonging to subcategories.
    """
    categories = {
        "declarations": {
            "functions": [],
            "variables": [], 
            "classes": [],
            "imports": [],
            "modules": [],
            "enums": []
        },
        "statements": {
            "expressions": [],
            "assignments": [],
            "loops": [],
            "conditionals": [],
            "returns": [],
            "exceptions": []
        },
        "expressions": {
            "calls": [],
            "literals": [],
            "identifiers": [],
            "binary_operations": [],
            "unary_operations": [],
            "member_access": []
        }
    }
    
    for node in ast_structure.get('nodes', []):
        node_type = (node.get('type') or "").lower()
        
        # Function declarations
        if any(keyword in node_type for keyword in ['function', 'method', 'def ', 'func ']):
            categories['declarations']['functions'].append(node['id'])
        
        # Class declarations
        elif any(keyword in node_type for keyword in ['class', 'struct', 'interface', 'trait', 'union']):
            categories['declarations']['classes'].append(node['id'])
        
        # Import statements
        elif any(keyword in node_type for keyword in ['import', 'include', 'using', 'require']):
            categories['declarations']['imports'].append(node['id'])

        # Module/namespace/package declarations
        elif any(keyword in node_type for keyword in ['module', 'namespace', 'package', 'export']):
            categories['declarations']['modules'].append(node['id'])

        # Enum declarations
        elif any(keyword in node_type for keyword in ['enum', 'enumeration']):
            categories['declarations']['enums'].append(node['id'])
        
        # Variable declarations
        elif any(keyword in node_type for keyword in ['declaration', 'definition', 'variable', 'var ']):
            categories['declarations']['variables'].append(node['id'])
        
        # Return statements
        elif 'return' in node_type:
            categories['statements']['returns'].append(node['id'])
        
        # Conditional statements
        elif any(keyword in node_type for keyword in ['if', 'switch', 'case', 'conditional']):
            categories['statements']['conditionals'].append(node['id'])
        
        # Loop statements
        elif any(keyword in node_type for keyword in ['for', 'while', 'loop']):
            categories['statements']['loops'].append(node['id'])

        # Exception handling statements
        elif any(keyword in node_type for keyword in ['try', 'except', 'catch', 'finally', 'throw', 'raise']):
            categories['statements']['exceptions'].append(node['id'])
        
        # Assignment statements
        elif any(keyword in node_type for keyword in ['assignment', 'assign']):
            categories['statements']['assignments'].append(node['id'])
        
        # Expression statements
        elif 'expression' in node_type:
            categories['statements']['expressions'].append(node['id'])
        
        # Call expressions
        elif any(keyword in node_type for keyword in ['call', 'invocation']):
            categories['expressions']['calls'].append(node['id'])
        
        # Binary operations
        elif any(keyword in node_type for keyword in ['binary', 'arithmetic', 'operator']):
            categories['expressions']['binary_operations'].append(node['id'])
        
        # Unary operations  
        elif 'unary' in node_type:
            categories['expressions']['unary_operations'].append(node['id'])

        # Member/field/property access
        elif any(keyword in node_type for keyword in ['member', 'field', 'property', 'access', 'dot']):
            categories['expressions']['member_access'].append(node['id'])
        
        # Identifiers
        elif any(keyword in node_type for keyword in ['identifier', 'name', 'symbol']):
            categories['expressions']['identifiers'].append(node['id'])
        
        # Literals
        elif any(keyword in node_type for keyword in ['literal', 'string', 'number', 'integer', 'float', 'boolean']):
            categories['expressions']['literals'].append(node['id'])
    
    return categories


def extract_name_from_text(text: str, language: str) -> str:
    """
    Heuristically extract a declaration or identifier name from a node text snippet.

    Args:
        text: Node text or snippet containing the declaration.
        language: Language string used to apply language-specific heuristics.

    Returns:
        Extracted name as a string, or "unknown" if extraction failed.
    """
    # Simple pattern matching for different languages
    if not text or not isinstance(text, str):
        return "unknown"
    if language == 'python':
        # Look for patterns like "def function_name" or "class ClassName"
        if 'def ' in text:
            return text.split('def ')[1].split('(')[0].strip()
        elif 'class ' in text:
            return text.split('class ')[1].split(':')[0].strip()
    elif language in ['c', 'cpp', 'java', 'c_sharp']:
        # Look for patterns like "void functionName" or "class ClassName"
        words = text.split()
        for i, word in enumerate(words):
            if word in ['class', 'struct', 'interface', 'union'] and i + 1 < len(words):
                return words[i + 1]
            elif i > 0 and words[i-1] in ['void', 'int', 'string', 'bool']:
                return word.split('(')[0]
    elif language in ['javascript', 'typescript']:
        # Look for patterns like "function functionName" or "class ClassName"
        if 'function ' in text:
            return text.split('function ')[1].split('(')[0].strip()
        elif 'class ' in text:
            return text.split('class ')[1].split('{')[0].split('extends')[0].strip()
        # Arrow functions: const name = () => or let name = function
        elif '=' in text and '=>' in text:
            return text.split('=')[0].strip().split()[-1]
        elif '=' in text and 'function' in text:
            return text.split('=')[0].strip().split()[-1]
    elif language == 'go':
        # Look for patterns like "func functionName" or "type StructName struct"
        if 'func ' in text:
            func_part = text.split('func ')[1]
            # Handle methods: func (receiver Type) methodName
            if func_part.strip().startswith('('):
                return func_part.split(')')[1].split('(')[0].strip()
            else:
                return func_part.split('(')[0].strip()
        elif 'type ' in text and 'struct' in text:
            return text.split('type ')[1].split('struct')[0].strip()
        elif 'type ' in text and 'interface' in text:
            return text.split('type ')[1].split('interface')[0].strip()
    elif language == 'ruby':
        # Look for patterns like "def method_name" or "class ClassName"
        if 'def ' in text:
            def_part = text.split('def ')[1]
            # Handle method names (stop at parentheses, newline, or end)
            for delimiter in ['(', '\n', '\r', ' ']:
                if delimiter in def_part:
                    return def_part.split(delimiter)[0].strip()
            return def_part.strip()
        elif 'class ' in text:
            class_part = text.split('class ')[1]
            # Stop at < (inheritance) or newline
            for delimiter in ['<', '\n', '\r']:
                if delimiter in class_part:
                    return class_part.split(delimiter)[0].strip()
            return class_part.strip()
        elif 'module ' in text:
            return text.split('module ')[1].split('\n')[0].strip()
    elif language == 'scala':
        # Look for patterns like "def methodName" or "class ClassName" or "object ObjectName"
        if 'def ' in text:
            def_part = text.split('def ')[1]
            # Handle generic types: def name[T]
            if '[' in def_part:
                return def_part.split('[')[0].strip()
            elif '(' in def_part:
                return def_part.split('(')[0].strip()
            else:
                return def_part.split(':')[0].split('=')[0].strip()
        elif 'class ' in text:
            class_part = text.split('class ')[1]
            # Stop at [, (, or extends
            for delimiter in ['[', '(', 'extends', '{']:
                if delimiter in class_part:
                    return class_part.split(delimiter)[0].strip()
            return class_part.strip()
        elif 'object ' in text:
            object_part = text.split('object ')[1]
            for delimiter in ['extends', '{', '\n']:
                if delimiter in object_part:
                    return object_part.split(delimiter)[0].strip()
            return object_part.strip()
        elif 'trait ' in text:
            return text.split('trait ')[1].split('[')[0].split('extends')[0].strip()
    
    return "unknown"


def create_simple_cross_language_map(ast_structure: Dict, categories: Dict, language: str) -> Dict:
    """
    Build a minimal cross-language map of functions, classes, and imports.

    Args:
        ast_structure: Serialized AST structure from extract_ast_structure.
        categories: Categorized node id lists from categorize_nodes_by_keywords.
        language: Normalized language name used for name extraction.

    Returns:
        Dict with keys 'function_declarations', 'class_declarations',
        and 'import_statements', each listing simple metadata dicts.
    """
    cross_map = {
        "function_declarations": [],
        "class_declarations": [], 
        "import_statements": []
    }
    
    # Map function declarations by analyzing node text
    for func_node_id in categories['declarations']['functions']:
        try:
            func_node = next((n for n in ast_structure['nodes'] if n['id'] == func_node_id), None)
            if not func_node:
                continue
            
            # Try to extract function name from text
            func_text = func_node.get('text', '')
            if not func_text:
                continue
                
            func_name = extract_name_from_text(func_text, language)
            
            function_info = {
                "node_id": func_node_id,
                "universal_type": "function",
                "name": func_name,
                "text_snippet": func_text[:100]  # First 100 chars
            }
            cross_map["function_declarations"].append(function_info)
        except Exception as e:
            print(f"Warning: Failed to process function node {func_node_id}: {e}")
            continue
    
    # Map class declarations
    for class_node_id in categories['declarations']['classes']:
        try:
            class_node = next((n for n in ast_structure['nodes'] if n['id'] == class_node_id), None)
            if not class_node:
                continue
            
            class_text = class_node.get('text', '')
            if not class_text:
                continue
                
            class_name = extract_name_from_text(class_text, language)
            
            class_info = {
                "node_id": class_node_id,
                "universal_type": "class", 
                "name": class_name,
                "text_snippet": class_text[:100]
            }
            cross_map["class_declarations"].append(class_info)
        except Exception as e:
            print(f"Warning: Failed to process class node {class_node_id}: {e}")
            continue
    
    # Map import statements
    for import_node_id in categories['declarations']['imports']:
        try:
            import_node = next((n for n in ast_structure['nodes'] if n['id'] == import_node_id), None)
            if not import_node:
                continue
            
            import_text = import_node.get('text', '')
            if not import_text:
                continue
                
            import_info = {
                "node_id": import_node_id,
                "text": import_text
            }
            cross_map["import_statements"].append(import_info)
        except Exception as e:
            print(f"Warning: Failed to process import node {import_node_id}: {e}")
            continue
    
    return cross_map

def convert_parquet_row_to_json(row: pd.Series) -> Optional[Dict]:
    """
    Convert a single parquet row (code sample) to the universal JSON schema.

    Args:
        row: Mapping-like row object containing at least 'language' and 'code'.
             Optional fields: 'line_count', 'avg_line_length'.

    Returns:
        JSON-serializable dict containing language, metadata, ast, and original_source_code,
        or None if conversion was skipped or failed.
    """
    try:
        # Expect columns: 'language', 'code', 'line_count', 'avg_line_length'
        language = row.get('language') if hasattr(row, "get") else row['language']
        code = row.get('code') if hasattr(row, "get") else row['code']
        if not language or not isinstance(language, str):
            logger.warning("Skipping row: missing or invalid 'language' field")
            return None
        if not code or not isinstance(code, str):
            logger.warning("Skipping row: missing or invalid 'code' field")
            return None

        # Handle language name variations
        language_mapping = {
            'c-sharp': 'c_sharp',
            'csharp': 'c_sharp',
            'c_sharp': 'c_sharp',
        }

        normalized_language = language_mapping.get(language, language)
        
        if normalized_language not in SUPPORTED_LANGUAGES:
            logger.info("Skipping unsupported language: %s -> %s", language, normalized_language)
            return None
            
        # Parse the code
        parser = get_parser(normalized_language)
        source_bytes = bytes(code, 'utf8')
        tree = parser.parse(source_bytes)
        
        # Extract AST structure
        ast_structure = extract_ast_structure(tree, code, normalized_language)
        
        # Add categorization
        node_categories = categorize_nodes_by_keywords(ast_structure, normalized_language)
        cross_language_map = create_simple_cross_language_map(ast_structure, node_categories, normalized_language)
        
        # Build result
        def count_categorized_nodes(cat_dict: Dict) -> int:
            total = 0
            for top_val in cat_dict.values():
                if isinstance(top_val, dict):
                    for sub_list in top_val.values():
                        if isinstance(sub_list, list):
                            total += len(sub_list)
                elif isinstance(top_val, list):
                    total += len(top_val)
            return total

        categorized_count = count_categorized_nodes(node_categories)

        result = {
            "language": normalized_language,
            "success": True,
            "metadata": {
                "lines": int(row.get('line_count')) if row.get('line_count') is not None else 0,
                "avg_line_length": float(row.get('avg_line_length')) if row.get('avg_line_length') is not None else 0.0,
                "nodes": len(ast_structure['nodes']),
                "errors": 0,
                "source_hash": hash_string(code),
                "categorized_nodes": categorized_count
            },
            "ast": ast_structure,
            "node_categories": node_categories,
            "cross_language_map": cross_language_map,
            "original_source_code": code,
        }
        
        logger.info("Converted row: language=%s nodes=%d categorized=%d",
                    normalized_language, len(ast_structure['nodes']), categorized_count)
        return result
        
    except Exception as e:
        try:
            lang_display = row.get('language', 'unknown') if hasattr(row, "get") else row['language']
        except Exception:
            lang_display = "unknown"
        logger.exception("Error converting row (language=%s): %s", lang_display, e)
        return None


def process_parquet_file(input_file: str, output_dir: str, max_rows: Optional[int] = None) -> Dict:
    """
    Read a parquet file, convert each row to the universal schema, and write results.

    Args:
        input_file: Path to input parquet file.
        output_dir: Directory where output parquet will be written.
        max_rows: Optional cap on number of rows to process from the input file.

    Returns:
        Summary dict with keys 'input_file', 'total_rows', and 'successful_conversions'.
    """
    logger.info("Reading parquet file: %s", input_file)
    df = pd.read_parquet(input_file)
    
    if max_rows:
        df = df.head(max_rows)

    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        json_data = convert_parquet_row_to_json(row)
        if json_data:
            results.append(json.dumps(json_data))
        else:
            results.append(None)

    assert len(results) == len(df), "Length not same. Something went wrong"

    df["universal_schema"] = results
    orig_length = len(df)
    df.dropna(inplace=True)
    successful_conversions = len(df)
    print("Rows lost after universal conversion: ", orig_length-successful_conversions)
    
    input_filename = Path(input_file).stem
    output_file = Path(output_dir) / f"{input_filename}.parquet"
    df.to_parquet(output_file)
    del df  # Free memory
    logger.info("Wrote converted parquet file: %s", output_file)
    
    return {
        "input_file": input_file,
        "total_rows": orig_length,
        "successful_conversions": successful_conversions
    }

def process_folder(input_folder: str, output_folder: str, max_rows_per_file: Optional[int] = None) -> None:
    """
    Process all parquet files in a folder, converting each and producing a summary.

    Args:
        input_folder: Directory containing input .parquet files.
        output_folder: Directory to write converted .parquet files.
        max_rows_per_file: Optional cap applied per-file when processing.

    Returns:
        None (prints a processing summary).
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    parquet_files = list(input_path.glob("*.parquet"))
    logger.info("Found %d parquet files in %s", len(parquet_files), input_folder)
    for file in parquet_files:
        print(f"  - {file.name}")
    
    all_stats = []
    for parquet_file in parquet_files:
        logger.info("Processing file: %s", parquet_file.name)
        stats = process_parquet_file(str(parquet_file), output_folder, max_rows_per_file)
        all_stats.append(stats)
        logger.info("Completed: %d/%d successful for %s",
                    stats['successful_conversions'], stats['total_rows'], Path(stats['input_file']).name)
    
    # Summary
    logger.info("=== PROCESSING SUMMARY ===")
    total_rows = sum(s['total_rows'] for s in all_stats)
    total_success = sum(s['successful_conversions'] for s in all_stats)
    
    for stats in all_stats:
        success_rate = stats['successful_conversions'] / stats['total_rows'] if stats['total_rows'] > 0 else 0
        logger.info("%s: %d/%d (%.1%%)", Path(stats['input_file']).name,
                    stats['successful_conversions'], stats['total_rows'], success_rate * 100)
    
    if total_rows > 0:
        overall_rate = total_success / total_rows
    else:
        overall_rate = 0.0

    logger.info("TOTAL: %d/%d (%.1f%% success rate)", total_success, total_rows, overall_rate * 100)


def main():
    INPUT_FOLDER = "parsed_data_parquet"  
    OUTPUT_FOLDER = Path.cwd() / "mlcpd_parquet"
    MAX_ROWS_PER_FILE = None

    # Create output folder
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    logger.info("Input folder: %s", INPUT_FOLDER)
    logger.info("Output folder: %s", OUTPUT_FOLDER)

    if not Path(INPUT_FOLDER).exists():
        logger.error("Input folder not found: %s", INPUT_FOLDER)
        exit(1)
            
    logger.info("Starting MLCPD Tree-Sitter Conversion...")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, MAX_ROWS_PER_FILE)
    logger.info("Done! Output parquet files in: %s", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()