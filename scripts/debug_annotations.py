from pathlib import Path
from vaast_app.utils.tree_viz_utils import TreeVizUtils


def debug_annotations():
    # Load tree and generate figure
    tree_viz_utils = TreeVizUtils(Path(__file__).parent / "full_collection_tree.nwk")
    tree = tree_viz_utils.get_collapsed_tree("phylum")
    fig = tree_viz_utils.generate_figure(tree)

    # Check annotations
    annotations = fig.layout.annotations
    print(f"Total annotations: {len(annotations)}")

    # Sample a few annotations from different quadrants
    # We can iterate and find ones with specific angles

    samples = []
    for ann in annotations:
        samples.append({"text": ann.text, "x": ann.x, "y": ann.y, "textangle": ann.textangle, "xanchor": ann.xanchor})

    # Sort by angle to see distribution
    samples.sort(key=lambda s: s["textangle"])

    print("\n--- Sample Annotations (Sorted by Angle) ---")
    step = max(1, len(samples) // 10)
    for i in range(0, len(samples), step):
        s = samples[i]
        print(f"Text: {s['text']}, Angle: {s['textangle']:.2f}, Anchor: {s['xanchor']}")


if __name__ == "__main__":
    debug_annotations()
