# MultiLang Code Parser Dataset (MLCPD)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-blue)](https://huggingface.co/datasets/jugalgajjar/MultiLang-Code-Parser-Dataset)


**MultiLang-Code-Parser-Dataset (MLCPD)** provides a large-scale, unified dataset of parsed source code across 10 major programming languages, represented under a universal schema that captures syntax, semantics, and structure in a consistent format.

Each entry corresponds to one parsed source file and includes:
- Language metadata
- Code-level statistics (lines, errors, AST nodes)
- Universal Schema JSON (normalized structural representation)

MLCPD enables robust cross-language analysis, code understanding, and representation learning by providing a consistent, language-agnostic data structure suitable for both traditional ML and modern LLM-based workflows.

---

## 📊 Key Statistics

| Metric | Value |
|--------|--------|
| Total Languages | 10 |
| Total Files | 40 |
| Total Records | 7,021,722 |
| Successful Conversions | 7,021,718 (99.9999%) |
| Failed Conversions | 4 (3 in C, 1 in C++) |
| Disk Size | ~105 GB (Parquet format) |
| Memory Size | ~580 GB (Parquet format) |

The dataset is clean, lossless, and statistically balanced across languages.  
It offers both per-language and combined cross-language representations.

---

## 🔍 Features

- **Universal Schema:** A unified structural representation harmonizing AST node types across languages.  
- **Compact Format:** Stored in Apache Parquet, allowing fast access and efficient querying.  
- **Cross-Language Compatibility:** Enables comparative code structure analysis across multiple programming ecosystems.  
- **Error-Free Parsing:** 99.9999% successful schema conversions across ~7M code files.  
- **Statistical Richness:** Includes per-language metrics such as mean line count, AST size, and error ratios.  
- **Ready for ML Pipelines:** Compatible with PyTorch, TensorFlow, Hugging Face Transformers, and graph-based models.

---

## 🚀 Use Cases

MLCPD can be directly used for:
- Cross-language code representation learning
- Program understanding and code similarity tasks
- Syntax-aware pretraining for LLMs
- Code summarization, clone detection, and bug prediction
- Graph-based learning on universal ASTs
- Benchmark creation for cross-language code reasoning

---

## 📥 How to Access the Dataset

### Using the Hugging Face `datasets` Library

This dataset is hosted on the Hugging Face Hub and can be easily accessed using the `datasets` library.

#### Install the Required Library

```bash
pip install datasets
```

#### Import Library

```bash
from datasets import load_dataset
```

#### Load the Entire Dataset

```bash
dataset = load_dataset(
    "jugalgajjar/MultiLang-Code-Parser-Dataset"
)
```

#### Load a Specific Language File

```bash
dataset = load_dataset(
    "jugalgajjar/MultiLang-Code-Parser-Dataset",
    data_files="python_parsed_1.parquet"
)
```

#### Stream Data

```bash
dataset = load_dataset(
    "jugalgajjar/MultiLang-Code-Parser-Dataset",
    data_files="python_parsed_1.parquet",
    streaming=True
)
```

#### Access Data Content (After Downloading)

```bash
try:
    for example in dataset["train"].take(5):
        print(example)
        print("-"*25)
except Exception as e:
    print(f"An error occurred: {e}")
```

### Manual Download

You can also manually download specific language files from the Hugging Face repository page:

1. Visit https://huggingface.co/datasets/jugalgajjar/MultiLang-Code-Parser-Dataset
2. Navigate to the Files tab
3. Click on the language file you want (e.g., `python_parsed_1.parquet`)
4. Use the Download button to save locally

---

## 🧩 AST Visualization Tool

A lightweight Python utility is included to visualize parsed Abstract Syntax Trees (ASTs) from the dataset in an interactive and hierarchical format.<br>
It works for all supported languages (e.g., Python, C++, Java, etc.) whose ASTs follow the dataset’s `universal_schema` structure.

### Overview

This tool renders each AST as a directed graph where:
- Nodes represent syntactic elements (functions, expressions, identifiers, etc.)
- Edges show parent–child relationships
- Colors distinguish root vs. nested nodes

### Usage

Run the script with:
```bash
python visualize_ast.py --json_parse example_ast.json --output ast_viz.html
```
where,
- `--json_parse` → Path to the JSON file which stores your AST
- `--output` → Output HTML filename (default: `ast_visualization.html`)

This generates an HTML visualization that can be opened in any browser:
```bash
open ast_viz.html
```

---

## 📂 Dataset Structure

```
MultiLang-Code-Parser-Dataset/
├── c_parsed_1.parquet
├── c_parsed_2.parquet
├── c_parsed_3.parquet
├── c_parsed_4.parquet
├── c_sharp_parsed_1.parquet
├── ...
└── typescript_parsed_4.parquet
```
Each file corresponds to one partition of a language (~175k rows each).

---

## 📜 License

This dataset is released under the [MIT License](/LICENSE).<br>
You are free to use, modify, and redistribute it for research and educational purposes, with proper attribution.

---

## 🙏 Acknowledgements

- [StarCoder Dataset](https://huggingface.co/datasets/bigcode/starcoderdata) for source code samples
- [TreeSitter](https://tree-sitter.github.io/tree-sitter/) for parsing
- [Hugging Face](https://huggingface.co/) for dataset hosting

---

## 📧 Contact

For questions, collaborations, or feedback:

- **Primary Author**: Jugal Gajjar
- **Email**: [812jugalgajjar@gmail.com](mailto:812jugalgajjar@gmail.com)
- **LinkedIn**: [linkedin.com/in/jugal-gajjar/](https://www.linkedin.com/in/jugal-gajjar/)

---

⭐ If you find this dataset useful, consider starring the repository and sharing your work that uses it.