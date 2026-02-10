from pathlib import Path
import sys

# Ensure vaast_app is in path
sys.path.append(str(Path.cwd()))

from vaast_app.utils.tree_viz_utils import TreeVizUtils


def test_tree_viz():
    nwk_path = Path("scripts/full_collection_tree.nwk")
    if not nwk_path.exists():
        print(f"Tree file not found at {nwk_path}")
        return

    utils = TreeVizUtils(nwk_path)

    print("Loading tree...")
    tree = utils.load_tree()
    print(f"Tree loaded. Root: {tree.name}, Descendants: {len(tree.get_descendants())}")

    print("Collapsing at 'class'...")
    collapsed = utils.get_collapsed_tree(target_rank="class")
    print(f"Collapsed tree leaves: {len(collapsed.get_leaves())}")

    # Check if a leaf has 'count' feature
    leaf = collapsed.get_leaves()[0]
    print(f"Sample leaf: {leaf.name}, Count: {getattr(leaf, 'count', 'N/A')}")

    print("Generating figure...")
    fig = utils.generate_figure(collapsed)
    print("Figure generated successfully.")

    # Optional: write figure to file to check?
    # fig.write_html("test_tree_viz.html")


if __name__ == "__main__":
    test_tree_viz()
