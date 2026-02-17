"""
This module contains utilities for generating radial tree visualizations.
It provides the TreeVizUtils class which handles tree loading, collapsing,
layout calculation (polar to cartesian), and Plotly figure generation.
"""

import math
from pathlib import Path
from typing import cast

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

# Rank list for index lookup
RANK_LIST = sorted(RANK_ORDER.keys(), key=lambda k: RANK_ORDER[k])


class NodeWithData(TreeNode):
    """
    Extended TreeNode class with additional attributes for visualization.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the node with default attributes for visualization.
        """
        super().__init__(*args, **kwargs)
        self.count = 0
        self.rank = ""
        self.sci_name = ""
        self.theta = 0.0
        self.r = 0.0


class TreeVizUtils:
    """
    Utilities for generating radial tree visualizations with collapsing at specific ranks.
    """

    @staticmethod
    def get_next_ranks(current_root_rank: str) -> tuple[str, str, str]:
        """
        Determine the next visualization configuration based on the new root rank.

        Rules:
          - Color/Label Rank = Root Rank + 1
          - Leaf Rank = Root Rank + 3

        Process:
          1. Initial View (Root=Bacteria/None):
             - Root Rank: Superkingdom (0)
             - Color Rank: Phylum (1)
             - Leaf Rank: Order (3)

          2. Zoom Level 1 (Root=Phylum):
             - Color Rank: Class (2)
             - Leaf Rank: Family (4)

        :param current_root_rank: The rank of the current root node (e.g., 'phylum').
        :return: A tuple containing (current_root_rank, color_rank, leaf_rank).
        """
        # Ensure rank is valid key
        if current_root_rank not in RANK_ORDER:
            # Default fallback
            return "superkingdom", "phylum", "order"

        current_idx = RANK_ORDER[current_root_rank]

        # Calculate indices
        color_idx = current_idx + 1
        leaf_idx = current_idx + 3

        # Cap at max rank
        max_idx = len(RANK_LIST) - 1
        color_idx = min(color_idx, max_idx)
        leaf_idx = min(leaf_idx, max_idx)

        color_rank = RANK_LIST[color_idx]
        leaf_rank = RANK_LIST[leaf_idx]

        return current_root_rank, color_rank, leaf_rank

    def __init__(self, nwk_path: Path):
        """
        Initialize the TreeVizUtils with the path to the Newick file.

        :param nwk_path: Path to the .nwk file containing the tree.
        """
        self.nwk_path = nwk_path
        self._full_tree: Tree | None = None

    def load_tree(self) -> Tree:
        """
        Load the full tree from the Newick file.

        The tree is cached after the first load. The method attempts to load with format=3
        (all internal nodes labeled) first, falling back to format=1 if necessary.

        :return: A copy of the loaded ete3 Tree object.
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

        Nodes at the target rank become the new leaves, and their children are removed.
        Descendant counts are added as features to the collapsed nodes.

        :param target_rank: The rank to collapse at (nodes at this rank become leaves).
        :param root_tax_name: Optional name of the root node to zoom into. If provided, the tree is rooted at this node.
        :return: A collapsed ete3 Tree.
        """
        tree = self.load_tree()

        # If a specific root is requested, find it and make it the new root
        if root_tax_name:
            # Search by name (scientific name or internal node name)
            # Use a robust search that handles spaces/underscores
            clean_name = root_tax_name.replace("_", " ")
            target_node = None
            for node in cast(TreeNode, tree.traverse()):
                if node.name == clean_name or getattr(node, "sci_name", "") == clean_name:
                    target_node = node
                    break

            if target_node:
                tree = target_node.detach()
            else:
                # Fallback: keep full tree if not found
                pass

        # Target rank logic
        target_rank_level = RANK_ORDER.get(target_rank.lower())
        if target_rank_level is None:
            return tree  # Invalid rank, return full tree

        # Collapse logic: Prune children of nodes at the target rank
        nodes_to_collapse = []
        target_rank_lower = target_rank.lower()

        for node in cast(TreeNode, tree.traverse("preorder")):
            node_rank = getattr(node, "rank", "")
            # If node matches rank, mark for collapse and stop traversing down this branch
            if str(node_rank).lower() == target_rank_lower:
                nodes_to_collapse.append(node)
                # No need to traverse deeper as children will be removed

        for node in nodes_to_collapse:
            # Calculate metadata before collapsing
            descendant_count = len(node.get_leaves())
            node.add_feature("count", descendant_count)
            node.children = []  # Remove children
            node.add_feature("collapsed", True)

        return tree

    @staticmethod
    def _generate_colors(n: int) -> list[str]:
        """
        Generate distinct colors using Golden Angle approximation.

        This method produces visually distinct colors by spacing hues evenly around the color wheel
        using the golden angle (approx. 137.5 degrees). It also varies saturation and lightness
        slightly to add further distinction.

        :param n: The number of colors to generate.
        :return: A list of HSL color strings.
        """
        colors = []
        for i in range(n):
            hue = (i * 137.508) % 360  # Golden angle
            # Vary saturation and lightness slightly to add variety
            saturation = 65 + (i % 3) * 10  # 65, 75, 85
            lightness = 45 + (i % 2) * 10  # 45, 55
            colors.append(f"hsl({hue:.1f}, {saturation}%, {lightness}%)")
        return colors

    @staticmethod
    def generate_figure(tree: Tree, color_by_rank: str | None = None, label_rank: str | None = None) -> go.Figure:
        """
        Generate a Plotly tree figure using Cartesian coordinates for custom styling.

        :param tree: The ete3 tree to visualize.
        :param color_by_rank: The rank to use for coloring sectors (e.g., 'phylum').
                              If None, colors are assigned per leaf.
        :param label_rank: The rank to use for labeling (e.g., 'phylum').
                           If None, leaves are labeled.
        """
        leaves = tree.get_leaves()
        num_leaves = len(leaves)
        if num_leaves == 0:
            return go.Figure()

        # 1. Calculate Angles (Polar)
        # User requested uniform slice sizes
        angle_per_leaf = 360.0 / num_leaves

        current_angle = 0.0
        max_dist = 0.0

        # Assign Angles and Radius to Leaves
        for leaf in leaves:
            width = angle_per_leaf
            start_angle = current_angle
            center_angle = start_angle + (width / 2)

            leaf.add_feature("theta", center_angle)
            leaf.add_feature("angle_width", width)
            leaf.add_feature("start_angle", start_angle)
            leaf.add_feature("end_angle", start_angle + width)

            dist = leaf.get_distance(tree)
            leaf.add_feature("r", dist)
            max_dist = max(max_dist, dist)

            current_angle += width

        # Assign Angles and Radius to Internal Nodes
        # We need this for drawing edges AND for finding the center of the 'color_by_rank' sectors
        for node in cast(TreeNode, tree.traverse("postorder")):
            if not node.is_leaf():
                children = node.children
                if children:
                    thetas = [child.theta for child in children]
                    node.add_feature("theta", sum(thetas) / len(thetas))

                    # For sectors, calculate the angular range
                    start_angles = [float(cast(float, getattr(c, "start_angle", c.theta))) for c in children]
                    end_angles = [float(cast(float, getattr(c, "end_angle", c.theta))) for c in children]
                    node.add_feature("start_angle", min(start_angles))
                    node.add_feature("end_angle", max(end_angles))

                    dist = node.get_distance(tree)
                    node.add_feature("r", dist)
                    max_dist = max(max_dist, dist)
                else:
                    node.add_feature("theta", 0)
                    node.add_feature("r", 0)

        # 2. Cartesian Conversion Helper
        def polar_to_cart(r, theta_deg):
            rad = math.radians(theta_deg)
            return r * math.cos(rad), r * math.sin(rad)

        # 3. Generate Traces
        fig = go.Figure()

        # --- Identify Groups for Coloring ---
        # If color_by_rank is specified, we want to group leaves by that ancestor.
        # We will collect the unique nodes at that rank.

        color_groups = []  # List of (Node, list[Leaves]) or similar

        if color_by_rank:
            # We need to find the specific ancestor for each leaf or traverse the tree to find nodes at rank
            rank_nodes = []
            for node in cast(TreeNode, tree.traverse()):
                if getattr(node, "rank", "") == color_by_rank:
                    rank_nodes.append(node)

            if not rank_nodes:
                # Fallback: treat leaves as the groups if rank not found
                rank_nodes = leaves

            # Now assign colors to these nodes
            palette = TreeVizUtils._generate_colors(len(rank_nodes))
            for i, node in enumerate(rank_nodes):
                color_groups.append((node, palette[i]))

        else:
            # Color every leaf individually
            palette = TreeVizUtils._generate_colors(len(leaves))
            for i, leaf in enumerate(leaves):
                color_groups.append((leaf, palette[i]))

        # --- A. Sectors (Background Pies) ---
        max_r_sector = max_dist * 1.05

        for node, color in color_groups:
            # Draw wedge for this node (covering all its descendants)
            # Use its start/end angle calculated during postorder

            theta_start = getattr(node, "start_angle", 0)
            theta_end = getattr(node, "end_angle", 0)

            # Handle wrap-around or 0-width (shouldn't happen with valid tree logic)
            if theta_end <= theta_start:
                if abs(theta_end - theta_start) < 1e-6:
                    continue

            # Interpolate arc
            # Resolution depends on width: ensure smoothness
            steps = max(2, int((theta_end - theta_start) / 1))
            wedge_x = [0]
            wedge_y = [0]

            for s in range(steps + 1):
                t = theta_start + (theta_end - theta_start) * (s / steps)
                wx, wy = polar_to_cart(max_r_sector, t)
                wedge_x.append(wx)
                wedge_y.append(wy)

            wedge_x.append(0)
            wedge_y.append(0)

            # Metadata for hover
            node_name = getattr(node, "sci_name", node.name)
            count = getattr(node, "count", len(node.get_leaves()))
            rank = getattr(node, "rank", "unknown")
            hover = f"{node_name}<br>Rank: {rank}<br>Descendants: {count}"

            fig.add_trace(
                go.Scatter(
                    x=wedge_x,
                    y=wedge_y,
                    fill="toself",
                    mode="lines",
                    line=dict(width=0, color=color),
                    name=node_name,
                    hoverinfo="text",
                    hovertext=hover,
                    customdata=[[node_name, rank, count]] * len(wedge_x),
                    showlegend=False,
                )
            )

        # --- B. Edges/Structure ---
        edge_x = []
        edge_y = []

        for node in cast(NodeWithData, tree.traverse()):
            if not node.is_root():
                parent = node.up

                # 1. Arc Segment: from Parent(theta) to self(theta) at Parent(r)
                # This draws the "horizontal" bar of the fork (which is an arc in polar)
                start_theta = parent.theta
                end_theta = node.theta
                radius = parent.r

                # Interpolate arc
                # Calculate delta and number of steps for smoothness
                delta_theta = end_theta - start_theta

                # Dynamic steps: at least 2 points, more for wider angles (approx 1 step per degree)
                steps = max(2, int(abs(delta_theta) / 1.0))

                for i in range(steps + 1):
                    # Linear interpolation of angle
                    t = start_theta + delta_theta * (i / steps)
                    ax, ay = polar_to_cart(radius, t)
                    edge_x.append(ax)
                    edge_y.append(ay)

                # 2. Radial Segment: from Parent(r) to self(r) at self(theta)
                # This draws the "vertical" drop of the fork (radial line)
                # The arc loop above ends at (radius, end_theta) = (parent.r, node.theta)
                # We extend from there to (node.r, node.theta)
                rx, ry = polar_to_cart(node.r, node.theta)
                edge_x.append(rx)
                edge_y.append(ry)

                # Add None to break the line between edges
                edge_x.append(None)
                edge_y.append(None)

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="black", width=0.5),
                hoverinfo="skip",  # Allow clicks to pass through to sectors
                showlegend=False,
            )
        )

        # --- C. Nodes (Hover Points) ---
        # Add interactive markers for all nodes (internal + leaves)
        node_x = []
        node_y = []
        node_customdata = []
        node_hover = []

        for node in cast(NodeWithData, tree.traverse()):
            # Skip root's marker? Usually root is at (0,0), we render it too.

            # Cartesian coordinates
            if node.is_root():
                nx, ny = 0, 0
            else:
                nx, ny = polar_to_cart(node.r, node.theta)

            node_x.append(nx)
            node_y.append(ny)

            # Metadata
            node_name = getattr(node, "sci_name", node.name)
            count = getattr(node, "count", len(node.get_leaves()))
            rank = getattr(node, "rank", "unknown")

            node_customdata.append([node_name, rank, count])
            node_hover.append(f"{node_name}<br>Rank: {rank}<br>Descendants: {count}")

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(size=5, color="black", opacity=0.3),
                name="Nodes",
                hoverinfo="text",
                hovertext=node_hover,
                customdata=node_customdata,
                showlegend=False,
            )
        )

        # --- D. Labels (Perimeter) ---
        annotations = []
        label_r = max_r_sector * 1.20

        # Decide which nodes to label
        nodes_to_label = []
        if label_rank:
            # Find nodes at this rank
            for node in cast(TreeNode, tree.traverse()):
                if getattr(node, "rank", "") == label_rank:
                    nodes_to_label.append(node)
            if not nodes_to_label:
                # If specifically requested rank not found (e.g. tree cut below it), do not label.
                pass
        else:
            # Label leaves
            nodes_to_label = leaves

        for node in nodes_to_label:
            start_angle = getattr(node, "start_angle", 0.0)
            end_angle = getattr(node, "end_angle", 0.0)
            theta = (start_angle + end_angle) / 2

            node_name = getattr(node, "sci_name", node.name)
            lx, ly = polar_to_cart(label_r + 0.02 * len(node_name), theta)

            # Rotation logic (Tangential)
            # We want the text to be perpendicular to the radius.
            norm_theta = theta % 360

            # Determine angle to keep text upright and readable
            if 0 <= norm_theta < 180:
                # Right Side / Top / Bottom-Right
                # Text runs "Down" at 3 o'clock? Or "Up"?
                # Usually text runs "Up" (-90 to 90).
                angle = 180 - norm_theta
                # Flip if needed for left-to-right reading
                if angle < -90:
                    angle += 180

                # Special cases near horizontal?
                # At 0 (Right), -90 (Down).
                # At 90 (Top), 0 (Horizontal).
                # At 180 (Left), 90 (Up).
                xanchor = "center"  # Tangential implies centered
            else:
                # Left Side / Bottom-Left
                angle = 180 - norm_theta
                # Flip 180 to be readable
                angle -= 180
                xanchor = "center"

            # Correction for specific quadrants to ensure consistency
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            if -90 <= norm_theta <= 0 or norm_theta >= 270:
                angle -= 180

            # Anchor tuning: Tangential text flows along the perimeter. Center is safest.
            # xanchor variable is already set above.

            annotations.append(
                dict(
                    x=lx,
                    y=ly,
                    text=node_name,
                    showarrow=False,
                    xanchor=xanchor,
                    yanchor="middle",
                    textangle=angle,
                    font=dict(size=10, color="black"),
                )
            )

        # --- E. Invisible Markers for Label Clicks ---
        # Annotations are not clickable. We place invisible markers at the label positions.
        label_x = []
        label_y = []
        label_customdata = []
        label_hover = []

        for node in nodes_to_label:
            start_angle = getattr(node, "start_angle", 0.0)
            end_angle = getattr(node, "end_angle", 0.0)
            theta = (start_angle + end_angle) / 2

            lx, ly = polar_to_cart(label_r, theta)

            node_name = getattr(node, "sci_name", node.name)
            count = getattr(node, "count", len(node.get_leaves()))
            rank = getattr(node, "rank", "unknown")

            label_x.append(lx)
            label_y.append(ly)
            label_customdata.append([node_name, rank, count])
            label_hover.append(f"{node_name}<br>Rank: {rank}<br>Descendants: {count}")

        fig.add_trace(
            go.Scatter(
                x=label_x,
                y=label_y,
                mode="markers",
                marker=dict(size=20, opacity=0),  # Invisible but clickable area
                name="Labels",
                hoverinfo="text",
                hovertext=label_hover,
                customdata=label_customdata,
                showlegend=False,
            )
        )

        # Layout styling
        limit = max_r_sector * 1.4

        fig.update_layout(
            xaxis=dict(visible=False, range=[-limit, limit], scaleanchor="y"),
            yaxis=dict(visible=False, range=[-limit, limit]),
            margin=dict(l=20, r=20, t=20, b=20),
            colorscale=dict(),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="closest",
            annotations=annotations,
        )

        return fig

    def process_interaction(
        self, triggered_id: str, click_data: dict | None, current_root_data: dict
    ) -> tuple[go.Figure, dict, str]:
        """
        Process user interaction (reset or click) and generate the updated tree figure.
        """
        new_root_name = current_root_data["name"]
        new_root_rank = current_root_data["rank"]

        # Handle callbacks
        if triggered_id == "reset-btn":
            new_root_name = "Bacteria"
            new_root_rank = "superkingdom"

        elif triggered_id == "tree-graph" and click_data:
            # Handle user click on the tree
            point = click_data["points"][0]
            # customdata is [node_name, rank, count]
            custom_data = point.get("customdata")
            if custom_data:
                clicked_name = custom_data[0]
                clicked_rank = custom_data[1]

                # Update root to clicked node
                # Only update if we can drill down further
                # Ensure rank is handled case-insensitively
                if isinstance(clicked_rank, list):
                    clicked_rank_lower = str(clicked_rank[0]).lower() if clicked_rank else ""
                else:
                    clicked_rank_lower = str(clicked_rank).lower() if clicked_rank else ""

                clicked_rank_idx = RANK_ORDER.get(clicked_rank_lower, 100)
                max_rank_idx = len(RANK_LIST) - 1

                if clicked_rank_idx < max_rank_idx:
                    new_root_name = clicked_name
                    new_root_rank = clicked_rank_lower
                else:
                    pass  # Cannot zoom in further

        # Calculate visualization parameters
        current_rank, color_rank, leaf_rank = self.get_next_ranks(new_root_rank)

        # Check if we are at the bottom
        if new_root_rank == leaf_rank:
            color_rank = None  # Color leaves individually

        # Generate Tree
        try:
            tree = self.get_collapsed_tree(leaf_rank, root_tax_name=new_root_name)
            fig = self.generate_figure(tree, color_by_rank=color_rank, label_rank=color_rank)
            fig.update_layout(title=f"Root: {new_root_name or 'Collection'} | Showing: {leaf_rank}")
        except Exception as e:
            # Fallback if something goes wrong (e.g. node not found)
            print(f"Error generating tree: {e}")
            fig = go.Figure()  # Empty

        breadcrumbs = f"Current Root: {new_root_name or 'All'} ({new_root_rank})"

        return fig, {"name": new_root_name, "rank": new_root_rank}, breadcrumbs
