from tree_sitter import Language

Language.build_library(
    "grammars/languages.so",
    [
        "tree_sitter_githubs/tree-sitter-c",
        "tree_sitter_githubs/tree-sitter-cpp",
        "tree_sitter_githubs/tree-sitter-c-sharp",
        "tree_sitter_githubs/tree-sitter-java",
        "tree_sitter_githubs/tree-sitter-javascript",
        "tree_sitter_githubs/tree-sitter-python",
        "tree_sitter_githubs/tree-sitter-ruby",
        "tree_sitter_githubs/tree-sitter-go",
        "tree_sitter_githubs/tree-sitter-scala",
        "tree_sitter_githubs/tree-sitter-typescript/typescript"
    ]
)