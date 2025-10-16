import json
import networkx as nx
from pyvis.network import Network
import argparse
import sys


def visualize_ast(ast_json, output_file="ast_visualization.html"):
    """
    Visualize an Abstract Syntax Tree (AST) as an interactive hierarchical graph.

    Args:
        ast_json (Union[str, list]): The parsed JSON string or list containing ASTs.
        output_file (str, optional): Output HTML filename for visualization.
                                     Defaults to "ast_visualization.html".

    Returns:
        None. Generates and saves an interactive HTML visualization.
    """
    if isinstance(ast_json, str):
        ast_json = json.loads(ast_json)

    G = nx.DiGraph()
    net = Network(height="840px", width="100%", directed=True,
                  bgcolor="#0a0a0a", font_color="white")

    # Traverse and add nodes
    for entry in ast_json:
        ast_data = entry.get("ast", {})
        nodes = ast_data.get("nodes", [])
        for node in nodes:
            label = f"{node['type']}\n({node['text'][:30]}...)" if len(node['text']) > 30 else f"{node['type']}\n{node['text']}"
            color = "#00c2ff" if node['parent'] is None else "#ffaa00"
            G.add_node(node["id"], label=label, color=color, shape="box", font={"size": 14})
            if node["parent"] is not None:
                G.add_edge(node["parent"], node["id"])

    # Convert to PyVis visualization
    net.from_nx(G)
    net.toggle_physics(True)
    # net.show_buttons(filter_=['physics'])
    net.write_html(output_file, notebook=False)
    print(f"AST visualization successfully saved to {output_file}")


def main():
    """
    Main execution point for AST visualization.
    """
    parser = argparse.ArgumentParser(
        description="Visualize an AST JSON file as an interactive graph."
    )
    parser.add_argument(
        "--json_parse",
        type=str,
        required=True,
        help="Path to the JSON file containing parsed ASTs."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ast_visualization.html",
        help="Output HTML file name for the visualization."
    )
    args = parser.parse_args()

    try:
        with open(args.json_parse, "r") as f:
            data = json.load(f)
        visualize_ast(data, args.output)
    except FileNotFoundError:
        print(f"File not found: {args.json_parse}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Ensure the file is valid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
