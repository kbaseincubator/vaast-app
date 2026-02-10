import math
from pathlib import Path
from typing import Any, Literal, cast

import plotly.graph_objects as go
from ete3 import Tree, TreeNode

# Rank order for determining hierarchy
RANK_ORDER = {
    "superkingdom": 0,
    "phylum": 1,
    "class": 2,
    "order": 3,
    "family": 4,
    "genus": 5,
    "species": 6,
    "subspecies": 7,
    "strain": 8,
}


class TreeVizUtils:
    """
    Utilities for generating radial tree visualizations with collapsing at specific ranks.
    """

    def __init__(self, nwk_path: Path):
        self.nwk_path = nwk_path
        self._full_tree: Tree | None = None

    def load_tree(self) -> Tree:
        """
        Load the full tree from the newick file. Cached.
        """
        if self._full_tree is None:
            # format=1 ensures internal node names are loaded, format=3 is more flexible
            # The generation script used format=3, quoted_node_names=True
            try:
                self._full_tree = Tree(str(self.nwk_path), format=3, quoted_node_names=True)
            except Exception:
                # Fallback or retry if format issue
                self._full_tree = Tree(str(self.nwk_path), format=1)
        return self._full_tree.copy()

    def get_collapsed_tree(self, target_rank: str = "class", root_tax_name: str | None = None) -> Tree:
        """
        Get a tree collapsed at the specified rank.

        :param target_rank: The rank to collapse at (nodes at this rank become leaves).
        :param root_tax_name: Optional name of the root node to zoom into.
        :return: A collapsed ete3 Tree.
        """
        tree = self.load_tree()

        # If a specific root is requested, find it and make it the new root
        if root_tax_name:
            # Search by name (scientific name or internal node name)
            # The newick format from generate_full_collection_tree might use taxids as names or sci_names
            # Let's try finding by name attribute
            clean_name = root_tax_name.replace("_", " ")  # robust to URL encoding
            target_node = None
            for node in tree.traverse():
                if node.name == clean_name or getattr(node, "sci_name", "") == clean_name:
                    target_node = node
                    break

            if target_node:
                tree = target_node.detach()
            else:
                # Fallback: keep full tree if not found (or raise error)
                pass

        # Target rank logic
        target_rank_level = RANK_ORDER.get(target_rank.lower())
        if target_rank_level is None:
            return tree  # invalid rank, return full tree

        # Collapse logic
        # We traverse effectively. If we hit a node at target_rank, we prune its children.
        # Note: Pre-order traversal is safe for this if we modify children list in place?
        # Actually proper way: iterate and mark nodes to keep/prune.

        nodes_to_collapse = []

        for node in tree.traverse("preorder"):
            node_rank = getattr(node, "rank", "")
            # If node matches rank, we stop traversing down
            if node_rank == target_rank:
                nodes_to_collapse.append(node)
                # We don't verify if children are lower rank, we just assume hierarchy holds or tree is noisy
                # For safety, we can check children's ranks? No, just collapse.

        for node in nodes_to_collapse:
            # Calculate metadata before collapsing
            descendant_count = len(node.get_leaves())
            node.add_feature("count", descendant_count)
            node.children = []  # Remove children
            node.add_feature("collapsed", True)

        return tree

    @staticmethod
    def generate_figure(tree: Tree) -> go.Figure:
        """
        Generate a Plotly radial tree figure from an ete3 Tree.
        """
        # 1. Calculate coordinates
        # We use a simple radial layout algorithm
        # theta (angle) from 0 to 360
        # r (radius) from 0 to max_depth

        leaves = tree.get_leaves()
        num_leaves = len(leaves)
        if num_leaves == 0:
            return go.Figure()

        # Assign angles to leaves
        angle_step = 360.0 / num_leaves
        for i, leaf in enumerate(leaves):
            leaf.add_feature("theta", i * angle_step)
            leaf.add_feature("r", leaf.get_distance(tree))  # Distance from root

        # Propagate angles to internal nodes (average of children)
        # And R is distance from root
        # We need postorder traversal (children processed first)
        for node in tree.traverse("postorder"):
            if not node.is_leaf():
                if node.children:
                    thetas = [child.theta for child in node.children]
                    # Handle 360 wraparound carefully? For simple trees, simple average is usually okay
                    # unless it spans 0/360 boundary.
                    # For now simple average.
                    node.add_feature("theta", sum(thetas) / len(thetas))
                    node.add_feature("r", node.get_distance(tree))
                else:
                    node.add_feature("theta", 0)
                    node.add_feature("r", 0)
            else:
                # Ensure r is set for leaves (already done above but ensuring consistency)
                node.add_feature("r", node.get_distance(tree))

        # 2. visual traces
        edge_x = []
        edge_y = []
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_customdata = []

        max_r = 0

        for node in tree.traverse():
            r = getattr(node, "r", 0)
            theta = getattr(node, "theta", 0)
            max_r = max(max_r, r)

            # Convert polar to cartesian for plotting?
            # Or use go.Scatterpolar. Scatterpolar is better for radial.

            # Add node
            node_x.append(r * math.cos(math.radians(theta)))  # Placeholder if cartesian
            # Actually for Scatterpolar we pass r and theta directly.

            # Store data for Scatterpolar
            # We construct edges as pairs of (r, theta) with None in between
            if not node.is_root():
                parent = node.up
                edge_x.extend([parent.r, node.r, None])  # r values
                edge_y.extend([parent.theta, node.theta, None])  # theta values

            # Node info
            node_name = getattr(node, "sci_name", node.name)
            count = getattr(node, "count", 0)
            rank = getattr(node, "rank", "unknown")
            hover = f"{node_name}<br>Rank: {rank}"
            if count:
                hover += f"<br>Descendants: {count}"

            node_text.append(hover)
            node_customdata.append([node_name, rank, count])

            # Color by rank or collapse status
            if getattr(node, "collapsed", False):
                node_colors.append("red")
            else:
                node_colors.append("blue")

        # Create traces
        # Edges
        edge_trace = go.Scatterpolar(
            r=edge_x, theta=edge_y, mode="lines", line=dict(color="gray", width=1), hoverinfo="none", showlegend=False
        )

        # Nodes
        # For actual nodes, we might want just leaves or all nodes?
        # Let's plot all nodes for now to show structure.
        node_r = []
        node_theta = []
        for node in tree.traverse():
            node_r.append(getattr(node, "r", 0))
            node_theta.append(getattr(node, "theta", 0))

        node_trace = go.Scatterpolar(
            r=node_r,
            theta=node_theta,
            mode="markers",
            marker=dict(
                size=[10 if getattr(n, "collapsed", False) else 5 for n in tree.traverse()],
                color=node_colors,
                line=dict(width=0),
            ),
            text=node_text,
            customdata=node_customdata,
            hoverinfo="text",
            showlegend=False,
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        return fig
