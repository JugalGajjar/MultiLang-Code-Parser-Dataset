import pandas as pd
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from tree_sitter import Parser, Node, Language


print("Loading compiled grammars from grammars/languages.so...")
LANGUAGE_OBJECTS = {}

languages_to_load = ['python', 'c', 'cpp', 'java', 'javascript', 'typescript', 'go', 'ruby', 'c_sharp', 'scala']
successful_loads = []

for lang in languages_to_load:
    try:
        lang_obj = Language('grammars/languages.so', lang)
        LANGUAGE_OBJECTS[lang] = lang_obj
        successful_loads.append(lang)
        print(f"✓ Loaded: {lang}")
    except Exception as e:
        print(f"✗ Failed to load {lang}: {e}")

SUPPORTED_LANGUAGES = {lang: lang for lang in successful_loads}
print(f"\nSupported languages: {list(SUPPORTED_LANGUAGES.keys())}")


def get_parser(language: str) -> Parser:
    """Get tree-sitter parser for specific language"""
    # Handle language name variations
    language_mapping = {
        'csharp': 'c_sharp',
        'c-sharp': 'c_sharp',
    }
    normalized_lang = language_mapping.get(language, language)
    
    if normalized_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language} -> {normalized_lang}")
    
    parser = Parser()
    parser.set_language(LANGUAGE_OBJECTS[normalized_lang])
    return parser


def test_all_parsers():
    """Test that ALL loaded language parsers work"""
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
                print(f"✓ {lang:12} parser works - root: {tree.root_node.type:20} children: {len(tree.root_node.children)}")
            except Exception as e:
                failed_parsers.append(lang)
                print(f"✗ {lang:12} parser failed: {e}")
    
    print(f"\nSummary: {len(working_parsers)} working, {len(failed_parsers)} failed")
    return working_parsers, failed_parsers

# Test all parsers
working_parsers, failed_parsers = test_all_parsers()


def hash_string(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def calculate_line_offsets(text: str) -> List[int]:
    offsets = [0]
    for i, char in enumerate(text):
        if char == '\n':
            offsets.append(i + 1)
    return offsets


def extract_ast_structure(tree, source_code: str, language: str) -> Dict:
    """
    Convert tree-sitter tree to structured nodes array using actual AST traversal
    with controlled granularity - filters out purely syntactic tokens
    """
    nodes = []
    node_id_counter = 0
    
    # Define tokens to skip (pure punctuation/delimiters)
    SKIP_TOKENS = {
        '(', ')', '{', '}', '[', ']', ',', ';', ':', 
        '.', '->', '::', '...', '|', '&'
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
        try:
            node_text = source_code[node.start_byte:node.end_byte].strip()
        except (IndexError, AttributeError):
            return False  # Keep the node if we can't extract text safely
        
        node_type = node.type
        
        # Skip if it's a pure punctuation/delimiter token
        if node_text in SKIP_TOKENS:
            return True
        
        # Skip if the node type itself is just punctuation
        if node_type in SKIP_NODE_TYPES:
            return True
        
        if node_type == "string" and (
            node_text.startswith('"""') or node_text.startswith("'''")
        ):
            return True

        if (
            node_type == "variable_declarator" 
            and node.parent is not None 
            and node.parent.type == "variable_declaration"
        ):
            try:
                parent_text = source_code[node.parent.start_byte:node.parent.end_byte]
                if node_text in parent_text:
                    return True
            except (IndexError, AttributeError):
                pass
        
        if (
            node_type == "type_spec" 
            and node.parent is not None 
            and node.parent.type == "type_declaration"
        ):
            try:
                parent_text = source_code[node.parent.start_byte:node.parent.end_byte]
                if node_text in parent_text:
                    return True
            except (IndexError, AttributeError):
                pass
        
        if (
            node_type == "variable_declarator" 
            and node.parent is not None 
            and node.parent.type == "lexical_declaration"
        ):
            try:
                parent_text = source_code[node.parent.start_byte:node.parent.end_byte]
                if node_text in parent_text:
                    return True
            except (IndexError, AttributeError):
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
        
        # Extract node text from source code
        node_text = source_code[node.start_byte:node.end_byte]

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
    Categorize nodes using keyword matching in node types (no field_name needed)
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
    
    for node in ast_structure['nodes']:
        node_type = node['type'].lower()
        
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
    """Extract name from node text using simple patterns"""
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
    Create cross-language mapping using node text analysis (no field_name needed)
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
    Convert a single parquet row to our tree-sitter JSON schema
    """
    try:
        # Handle language name variations
        language = row['language']
        language_mapping = {
            'c-sharp': 'c_sharp',
            'csharp': 'c_sharp',
            'c_sharp': 'c_sharp',
        }

        normalized_language = language_mapping.get(language, language)
        
        if normalized_language not in SUPPORTED_LANGUAGES:
            print(f"Skipping: {language} -> {normalized_language} not supported")
            return None
            
        # Parse the code
        parser = get_parser(normalized_language)
        source_bytes = bytes(row['code'], 'utf8')
        tree = parser.parse(source_bytes)
        
        # Extract AST structure
        ast_structure = extract_ast_structure(tree, row['code'], normalized_language)
        
        # Add categorization
        node_categories = categorize_nodes_by_keywords(ast_structure, normalized_language)
        cross_language_map = create_simple_cross_language_map(ast_structure, node_categories, normalized_language)
        
        # Build result
        result = {
            "language": normalized_language,
            "success": True,
            "metadata": {
                "lines": int(row['line_count']),
                "avg_line_length": float(row['avg_line_length']),
                "nodes": len(ast_structure['nodes']),
                "errors": 0,
                "source_hash": hash_string(row['code']),
                "categorized_nodes": sum(len(cats) for cats in node_categories.values())
            },
            "ast": ast_structure,
            "original_source_code": row['code'],
        }
        
        print(f"✓ {normalized_language}: {len(ast_structure['nodes'])} nodes, {result['metadata']['categorized_nodes']} categorized")
        return result
        
    except Exception as e:
        print(f"✗ Error converting {row.get('language', 'unknown')}: {e}...")
        return None


def process_parquet_file(input_file: str, output_dir: str, max_rows: Optional[int] = None) -> Dict:
    """Process a single parquet file"""
    print(f"Reading {input_file}...")
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
    print("Rows lost after universal conversion: ", orig_length-len(df))
    
    input_filename = Path(input_file).stem
    output_file = Path(output_dir) / f"{input_filename}.parquet"
    df.to_parquet(output_file)
    
    return {
        "input_file": input_file,
        "total_rows": orig_length,
        "successful_conversions": len(df)
    }

def process_folder(input_folder: str, output_folder: str, max_rows_per_file: Optional[int] = None) -> None:
    """Main processing function"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    parquet_files = list(input_path.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files:")
    for file in parquet_files:
        print(f"  - {file.name}")
    
    all_stats = []
    for parquet_file in parquet_files:
        print(f"\n{'='*50}")
        print(f"Processing {parquet_file.name}...")
        stats = process_parquet_file(str(parquet_file), output_folder, max_rows_per_file)
        all_stats.append(stats)
        print(f"Completed: {stats['successful_conversions']}/{stats['total_rows']} successful")
    
    # Summary
    print(f"\n{'='*50}")
    print("PROCESSING SUMMARY")
    print(f"{'='*50}")
    total_rows = sum(s['total_rows'] for s in all_stats)
    total_success = sum(s['successful_conversions'] for s in all_stats)
    
    for stats in all_stats:
        success_rate = stats['successful_conversions'] / stats['total_rows'] if stats['total_rows'] > 0 else 0
        print(f"{Path(stats['input_file']).name}: {stats['successful_conversions']}/{stats['total_rows']} ({success_rate:.1%})")
    
    print(f"{'='*50}")
    print(f"TOTAL: {total_success}/{total_rows} ({total_success/total_rows:.1%} success rate)")

INPUT_FOLDER = "testing_big"  
OUTPUT_FOLDER = Path.cwd() / "testing_big_out"
MAX_ROWS_PER_FILE = None

# Create output folder
OUTPUT_FOLDER.mkdir(exist_ok=True)
print(f"Input: {INPUT_FOLDER}")
print(f"Output: {OUTPUT_FOLDER}")

# === RUN PROCESSING ===
if __name__ == "__main__":
    if not Path(INPUT_FOLDER).exists():
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        exit(1)
        
    print("Starting MLCPD Tree-Sitter Conversion...")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, MAX_ROWS_PER_FILE)
    print(f"\nDone! JSON files in: {OUTPUT_FOLDER}")