# MultiLang Code Parser Dataset (MLCPD)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-blue)](https://huggingface.co/datasets/jugalgajjar/Filtered-StarCoder-Dataset-Mini)

A comprehensive multi-language code dataset that will be processed into a parsing benchmark for language-agnostic AI code parsers.

## Current Status: Filtered StarCoder Dataset Mini

This is the first phase of the MLCPD project - a cleaned and filtered version of code samples from the StarCoder dataset. The next phase will involve parsing these samples to create the full MultiLang Code Parser Dataset.

### Key Features (Current Dataset)

- **Cleaned and Filtered Code**: Samples have been processed to remove outliers in terms of line length and code size
- **Quality Metrics**: Each sample includes metadata about average line length and line count
- **Multi-language Support**: 10 programming languages represented in separate subsets
- **Consistent Format**: All samples follow the same Parquet structure for easy processing

## Dataset Statistics (Current)

| Language   | Sample Count | Avg. Line Length | Avg. Line Count |
|------------|--------------|------------------|-----------------|
| C          | 1,752,078    | 22.28           | 74.04            |
| C++        | 1,769,333    | 23.33           | 103.07           |
| C#         | 1,763,508    | 25.40           | 51.04            |
| Go         | 1,751,120    | 20.51           | 81.30            |
| Java       | 1,779,659    | 25.08           | 64.11            |
| JavaScript | 1,718,133    | 23.04           | 50.74            |
| Python     | 1,764,099    | 26.29           | 65.67            |
| Ruby       | 1,756,771    | 21.87           | 33.38            |
| Scala      | 952,890      | 27.97           | 53.44            |
| TypeScript | 1,738,885    | 23.82           | 42.90            |

## Future Vision: Full MLCPD

The complete MultiLang Code Parser Dataset will include:

- Unified Abstract Syntax Tree (AST) representation in a standard JSON/Parquet format
- Complete parsing metadata and language identification
- Benchmarking framework for AI parser evaluation

### Planned Data Format

```json
{
  "language": "python",
  "success": true,
  "metadata": {
    "filename": "example.py",
    "avg_line_length": 34.46,
    "line_count": 69,
    "ast_node_count": 112
  },
  "ast": {
    "type": "Module",
    "body": [
      {
        "type": "ImportFrom",
        "module": "math",
        "names": [
          {
            "type": "alias",
            "name": "sqrt",
            "asname": null
          }
        ],
        "level": 0
      }
      // Additional AST nodes...
    ]
  }
}
```

## How to Access the Current Dataset

### Using the Hugging Face `datasets` Library

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Load entire dataset
dataset = load_dataset("jugalgajjar/Filtered-StarCoder-Dataset-Mini")

# Load specific language
python_dataset = load_dataset(
    "jugalgajjar/Filtered-StarCoder-Dataset-Mini",
    data_files="python.parquet"
)
```

### Manual Download

1. Visit [dataset page](https://huggingface.co/datasets/jugalgajjar/Filtered-StarCoder-Dataset-Mini)
2. Navigate to "Files" tab
3. Download desired language files (e.g., `python.parquet`)

## Dataset Creation Process

1. **Data Collection**: Code samples from StarCoder dataset
2. **Filtering**: Removed outliers using IQR method
3. **Normalization**: Standardized format across languages
4. **Metadata Generation**: Calculated line metrics
5. **Serialization**: Stored in Parquet format  

*(Future steps will include parsing and AST conversion)*

## Citation

```bibtex
@misc{fscdmini2025,
  author = {Jugal Gajjar, Kamalasankari Subramaniakuppusamy, Kaustik Ranaware},
  title = {Filtered StarCoder Dataset Mini},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/jugalgajjar/Filtered-StarCoder-Dataset-Mini}}
}
```

*(Will be updated when full MLCPD is released)*

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [StarCoder Dataset](https://huggingface.co/datasets/bigcode/starcoderdata) for source code samples
- [TreeSitter](https://tree-sitter.github.io/tree-sitter/) for future parsing work
- [Hugging Face](https://huggingface.co/) for dataset hosting
