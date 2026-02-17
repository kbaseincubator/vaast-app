from pathlib import Path
from vaast_app.utils.tree_viz_utils import TreeVizUtils, RANK_ORDER


def test_logic():
    nwk_path = Path(__file__).parent / "full_collection_tree.nwk"
    if not nwk_path.exists():
        print("Skipping test: full_collection_tree.nwk not found")
        return

    utils = TreeVizUtils(nwk_path)

    print("--- Test 1: Initial View (Root=Bacteria, Color=Phylum, Leaf=Order) ---")
    # Rule from develop_tree_explorer:
    # Initial: Root=Bacteria, Color=Phylum, Leaf=Order

    # develop_tree_explorer now defaults to "Bacteria"
    tree_1 = utils.get_collapsed_tree("order", root_tax_name="Bacteria")
    leaves_1 = tree_1.get_leaves()
    print(f"Number of leaves (Order level) for Bacteria: {len(leaves_1)}")

    # Check if root is Bacteria
    root_name = getattr(tree_1, "name", "")
    print(f"Root Name: {root_name}")
    # Note: Depending on tree source, name might be taxid or sci_name.
    # TreeVizUtils.get_collapsed_tree detaches the node, so tree_1 is the Bacteria node.
    # Check sci_name if available
    root_sci_name = getattr(tree_1, "sci_name", root_name)
    print(f"Root Sci Name: {root_sci_name}")
    assert "Bacteria" in (root_name, root_sci_name), f"Expected root 'Bacteria', got {root_name}/{root_sci_name}"

    # Check rank of a random leaf
    sample_leaf = leaves_1[0]
    print(f"Sample leaf rank: {getattr(sample_leaf, 'rank', 'Unknown')}")
    assert (
        getattr(sample_leaf, "rank", "") == "order"
    ), f"Expected leaf rank 'order', got {getattr(sample_leaf, 'rank', '')}"

    # Generate figure to ensure no crash
    fig_1 = utils.generate_figure(tree_1, color_by_rank="phylum", label_rank="phylum")
    print("Figure 1 generated successfully.")

    print("\n--- Test 2: Zoom to Phylum (e.g. Proteobacteria) ---")
    # Simulate clicking 'Proteobacteria'
    # Find Proteobacteria node name
    # We need to find a phylum node.
    # In tree_1, Phylum nodes are internal.

    # Let's search in the full tree for a valid Phylum to click
    full_tree = utils.load_tree()
    target_phylum = None
    for node in full_tree.traverse():
        if getattr(node, "rank", "") == "phylum":
            target_phylum = node
            break

    if not target_phylum:
        print("No Phylum found to test zoom.")
        return

    phylum_name = getattr(target_phylum, "sci_name", target_phylum.name)
    print(f"Clicking Phylum: {phylum_name}")

    # Logic from develop_tree_explorer for Zoom 1:
    # Root = Phylum
    # Color = Class
    # Leaf = Family

    tree_2 = utils.get_collapsed_tree("family", root_tax_name=phylum_name)
    leaves_2 = tree_2.get_leaves()
    print(f"Number of leaves (Family level) in {phylum_name}: {len(leaves_2)}")

    if len(leaves_2) > 0:
        sample_leaf_2 = leaves_2[0]
        print(f"Sample leaf rank: {getattr(sample_leaf_2, 'rank', 'Unknown')}")
        assert (
            getattr(sample_leaf_2, "rank", "") == "family"
        ), f"Expected leaf rank 'family', got {getattr(sample_leaf_2, 'rank', '')}"

    fig_2 = utils.generate_figure(tree_2, color_by_rank="class", label_rank="class")
    print("Figure 2 generated successfully.")

    print("\n--- Test 2: Zoom to Phylum (e.g. Proteobacteria) ---")
    # ... (existing code) ...
    print("\n--- Verification Passed ---")


def test_rank_logic():
    print("\n--- Test 3: Rank Logic (get_next_ranks) ---")
    # 1. Bacteria (Superkingdom) -> Phylum, Order
    curr, color, leaf = TreeVizUtils.get_next_ranks("superkingdom")
    print(f"Superkingdom -> Color: {color}, Leaf: {leaf}")
    assert color == "phylum", f"Expected phylum, got {color}"
    assert leaf == "order", f"Expected order, got {leaf}"

    # 2. Phylum -> Class, Family
    curr, color, leaf = TreeVizUtils.get_next_ranks("phylum")
    print(f"Phylum -> Color: {color}, Leaf: {leaf}")
    assert color == "class", f"Expected class, got {color}"
    assert leaf == "family", f"Expected family, got {leaf}"

    # 3. Genus -> Species, Strain (Max)
    # Genus is 5. Color=Species(6), Leaf=Strain(8)
    curr, color, leaf = TreeVizUtils.get_next_ranks("genus")
    print(f"Genus -> Color: {color}, Leaf: {leaf}")
    assert color == "species", f"Expected species, got {color}"
    assert leaf == "strain", f"Expected strain, got {leaf}"

    # 4. Strain (Leaf) -> Strain, Strain (Max cap)
    curr, color, leaf = TreeVizUtils.get_next_ranks("strain")
    print(f"Strain -> Color: {color}, Leaf: {leaf}")
    assert color == "strain", f"Expected strain, got {color}"
    assert leaf == "strain", f"Expected strain, got {leaf}"

    print("Rank logic verified.")


def test_interaction_logic():
    print("\n--- Test 4: Interaction Logic (process_interaction) ---")
    nwk_path = Path(__file__).parent / "full_collection_tree.nwk"
    if not nwk_path.exists():
        print("Skipping interaction test: tree not found")
        return

    utils = TreeVizUtils(nwk_path)

    # 1. Initial Load (triggered_id="initial")
    # Current root: Bacteria
    current_root = {"name": "Bacteria", "rank": "superkingdom"}
    fig, new_root, breadcrumbs = utils.process_interaction("initial", None, current_root)
    print(f"Initial -> New Root: {new_root['name']} ({new_root['rank']})")
    assert new_root["name"] == "Bacteria"
    assert new_root["rank"] == "superkingdom"

    # 2. Click Phylum (e.g. Pseudomonadota)
    # Search for a valid phylum node
    full_tree = utils.load_tree()
    target_node = None
    for node in full_tree.traverse():
        if getattr(node, "rank", "") == "phylum":
            target_node = node
            break

    if target_node:
        t_name = getattr(target_node, "sci_name", target_node.name)
        t_rank = "phylum"
        print(f"Simulating click on: {t_name} ({t_rank})")

        click_data = {"points": [{"customdata": [t_name, t_rank, 100]}]}

        fig, new_root, breadcrumbs = utils.process_interaction("tree-graph", click_data, current_root)
        print(f"Click Phylum -> New Root: {new_root['name']} ({new_root['rank']})")

        assert new_root["name"] == t_name
        assert new_root["rank"] == "phylum"

    # 3. Reset
    fig, new_root, breadcrumbs = utils.process_interaction("reset-btn", None, current_root)
    print(f"Reset -> New Root: {new_root['name']} ({new_root['rank']})")
    assert new_root["name"] == "Bacteria"
    assert new_root["rank"] == "superkingdom"

    print("Interaction logic verified.")


if __name__ == "__main__":
    test_logic()
    test_rank_logic()
    test_interaction_logic()
