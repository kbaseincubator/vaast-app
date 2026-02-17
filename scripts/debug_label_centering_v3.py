from pathlib import Path
from vaast_app.utils.tree_viz_utils import TreeVizUtils


def main():
    nwk_path = Path(__file__).parent / "full_collection_tree.nwk"
    if not nwk_path.exists():
        print("Tree file not found.")
        return

    print("Initializing TreeVizUtils...")
    utils = TreeVizUtils(nwk_path)

    print("Collapsing tree...")
    tree = utils.get_collapsed_tree("order")

    # Generate figure to trigger calculations on nodes
    print("Generating figure to populate angles...")
    fig = utils.generate_figure(tree, label_rank="phylum")

    # Find key nodes
    targets = [
        "Pseudomonadota",
        "Chloroflexota",
        "Mycoplasmatota",
        "Deinococcota",
        "Actinomycetota",
    ]  # Pseudomonadota is near 60, Chloroflexota 170.

    # Let's check angles manually from known positions if possible.
    # But checking actual figure annotations is best.

    annotations = fig.layout.annotations

    for ann in annotations:
        text = getattr(ann, "text", "")
        for t in targets:
            if t in text:
                print(f"\n--- {t} Debug Info ---")
                print(f"Text Angle: {getattr(ann, 'textangle', 'N/A')}")
                print(f"X Anchor: {getattr(ann, 'xanchor', 'N/A')}")


if __name__ == "__main__":
    main()
