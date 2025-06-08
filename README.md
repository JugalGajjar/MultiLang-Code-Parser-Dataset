# MultiLang Code Parser Dataset (MLCPD)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-blue)](https://huggingface.co/datasets/jugalgajjar/MultiLang-Code-Parser-Dataset)

A comprehensive multi-language code dataset that will be processed into a parsing benchmark for language-agnostic AI code parsers.

## Current Status: Filtered StarCoder Dataset Mini

This is the second phase of the MLCPD project - the cleaned and filtered version of code samples from the StarCoder dataset has been parsed with language-specific parsers using tree-sitter. The next phase will involve converting these outputs to a unified JSON format to create the full MultiLang Code Parser Dataset.

### Key Features (Current Dataset)

- **Cleaned and Filtered Code**: Samples have been processed to remove outliers in terms of line length and code size
- **Quality Metrics**: Each sample includes metadata about average line length and line count of code along with AST node count and error count
- **Multi-language Support**: 10 programming languages represented in separate subsets
- **Consistent Format**: All samples follow the same Parquet structure for easy processing

## Dataset Statistics (Current)

| Language   | Sample Count | Avg. Line Length | Avg. Line Count |
|------------|--------------|------------------|-----------------|
| C          | 700,821      | 28.08           | 61.76            |
| C++        | 707,641      | 28.16           | 87.88            |
| C#         | 705,203      | 29.53           | 44.26            |
| Go         | 700,331      | 25.18           | 68.22            |
| Java       | 711,922      | 30.85           | 54.40            |
| JavaScript | 687,775      | 27.69           | 44.15            |
| Python     | 706,126      | 32.67           | 54.70            |
| Ruby       | 703,473      | 27.35           | 27.41            |
| Scala      | 702,833      | 35.30           | 44.38            |
| TypeScript | 695,597      | 29.18           | 36.89            |

## Future Vision: Full MLCPD

The complete MultiLang Code Parser Dataset will include:

- Unified Abstract Syntax Tree (AST) representation in a standard JSON/Parquet format
- Complete parsing metadata and language identification
- Benchmarking framework for AI parser evaluation

### Planned Data Format

#### Code
```python
class Dog:
    def init(self, name, breed):
        self.name = name
        self.breed = breed
    def bark(self):
        return f"{self.name} says Woof!"
def greet_dog(dog):
    return f"Hello, {dog.name}!"
# Create a Dog object
my_dog = Dog("Buddy", "Golden Retriever")
# Call functions
print(greet_dog(my_dog))
print(my_dog.bark())
```

#### Parsed Output
```json
{
  "language": "python",
  "success": true,
  "metadata": {
    "lines": 12,
    "nodes": 45,
    "errors": 0
  },
  "imports": [],
  "functions": [
    {
      "name": "greet_dog",
      "params": ["dog"],
      "body": [
        {
          "type": "return",
          "value": "f\"Hello, {dog.name}!\""
        }
      ]
    }
  ],
  "classes": [
    {
      "name": "Dog",
      "methods": [
        {
          "name": "__init__",
          "params": ["self", "name", "breed"],
          "body": [
            {
              "type": "assign",
              "target": "self.name",
              "value": "name"
            },
            {
              "type": "assign",
              "target": "self.breed", 
              "value": "breed"
            }
          ]
        },
        {
          "name": "bark",
          "params": ["self"],
          "body": [
            {
              "type": "return",
              "value": "f\"{self.name} says Woof!\""
            }
          ]
        }
      ]
    }
  ],
  "variables": [],
  "main_body": [
    {
      "type": "assign",
      "target": "my_dog",
      "value": {
        "type": "constructor",
        "class": "Dog",
        "args": ["\"Buddy\"", "\"Golden Retriever\""]
      }
    },
    {
      "type": "call",
      "name": "print",
      "args": [
        {
          "type": "call",
          "name": "greet_dog",
          "args": ["my_dog"]
        }
      ]
    },
    {
      "type": "call",
      "name": "print", 
      "args": [
        {
          "type": "method_call",
          "object": "my_dog",
          "method": "bark",
          "args": []
        }
      ]
    }
  ]
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
dataset = load_dataset("jugalgajjar/MultiLang-Code-Parser-Dataset")

# Load specific language
python_dataset = load_dataset(
    "jugalgajjar/MultiLang-Code-Parser-Dataset",
    data_files="python_parsed_1.parquet"
)
```

### Manual Download

1. Visit [dataset page](https://huggingface.co/datasets/jugalgajjar/MultiLang-Code-Parser-Dataset)
2. Navigate to "Files" tab
3. Download desired language files (e.g., `python_parsed_1.parquet`)

## Dataset Creation Process

1. **Data Collection**: Code samples from StarCoder dataset
2. **Filtering**: Removed outliers using IQR method
3. **Normalization**: Standardized format across languages
4. **Metadata Generation**: Calculated line metrics
5. **Serialization**: Stored in Parquet format
6. **Tree Sitter Parsing**: Parsed code using language-specific parsers

*(Future steps will include AST conversion)*

## Citation

```bibtex
@misc{fscdmini2025,
  author = {Jugal Gajjar, Kamalasankari Subramaniakuppusamy, Kaustik Ranaware},
  title = {Filtered StarCoder Dataset Mini},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/jugalgajjar/MultiLang-Code-Parser-Dataset}}
}
```

*(Will be updated when full MLCPD is released)*

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [StarCoder Dataset](https://huggingface.co/datasets/bigcode/starcoderdata) for source code samples
- [TreeSitter](https://tree-sitter.github.io/tree-sitter/) for parsing
- [Hugging Face](https://huggingface.co/) for dataset hosting