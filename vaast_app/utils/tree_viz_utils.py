import math
from pathlib import Path

import plotly.graph_objects as go
from ete3 import Tree

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


class TreeVizUtils:
    """
    Utilities for generating radial tree visualizations with collapsing at specific ranks.
    """

    @staticmethod
    def get_next_ranks(current_root_rank: str) -> tuple[str, str, str]:
        """
        Determine the next configuration based on the new root rank.
        Rule:
          - Color/Label Rank = Root Rank + 1
          - Leaf Rank = Root Rank + 3  (approx "render down 2 levels" from color group?)

          Initial (Root=None/Bacteria):
             Root Rank ~ Superkingdom (0)
             Color Rank = Phylum (1)
             Leaf Rank = Order (3)

          Zoom 1 (Root=Phylum):
             Color Rank = Class (2)
             Leaf Rank = Family (4)
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

        target_rank_lower = target_rank.lower()

        for node in tree.traverse("preorder"):
            node_rank = getattr(node, "rank", "")
            # If node matches rank, we stop traversing down
            if str(node_rank).lower() == target_rank_lower:
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
    def _generate_colors(n: int) -> list[str]:
        """Generate distinct colors using Golden Angle approximation."""
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
        for node in tree.traverse("postorder"):
            if not node.is_leaf():
                children = node.children
                if children:
                    thetas = [child.theta for child in children]
                    node.add_feature("theta", sum(thetas) / len(thetas))

                    # For sectors, we might want the range
                    start_angles = [float(getattr(c, "start_angle", c.theta)) for c in children]
                    end_angles = [float(getattr(c, "end_angle", c.theta)) for c in children]
                    # This simple min/max works if we don't cross 0/360 boundary in a messy way
                    # But since we increment form 0 to 360, it should be fine for contiguous groups
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
            # Traversing the tree is safer to find the 'sectors' directly.
            rank_nodes = []
            for node in tree.traverse():
                if getattr(node, "rank", "") == color_by_rank:
                    rank_nodes.append(node)

            # If no nodes found at that rank (e.g. tree is cut below phylum), fall back to leaves?
            # Or maybe the root is the rank?
            if not rank_nodes:
                # Fallback: treat leaves as the groups
                rank_nodes = leaves

            # Now assign colors to these nodes
            palette = TreeVizUtils._generate_colors(len(rank_nodes))
            # Create a map for quick lookup if needed, or just iterate these nodes to draw sectors
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

            # Handle wrap-around or 0-width?
            if theta_end <= theta_start:
                # Should not happen for valid tree with width
                # But if 360 reached, might be issue?
                # floating point tolerance?
                if abs(theta_end - theta_start) < 1e-6:
                    continue

            # Interpolate arc
            # Resolution depends on width
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

        for node in tree.traverse():
            if not node.is_root():
                parent = node.up
                px, py = polar_to_cart(parent.r, parent.theta)
                nx, ny = polar_to_cart(node.r, node.theta)
                edge_x.extend([px, nx, None])
                edge_y.extend([py, ny, None])

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

        for node in tree.traverse():
            # Skip root if it's just a container? No, root might be actionable.

            # Cartesian coordinates
            # Note: root might have r=0, theta=0
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
        label_r = max_r_sector * 1.02

        # Decide which nodes to label
        nodes_to_label = []
        if label_rank:
            # Find nodes at this rank
            for node in tree.traverse():
                if getattr(node, "rank", "") == label_rank:
                    nodes_to_label.append(node)
            if not nodes_to_label:
                # If specifically requested rank not found, maybe don't label anything?
                # Or fall back to leaves? Let's label nothing to avoid clutter if hierarchy doesn't match.
                pass
        else:
            # Label leaves
            nodes_to_label = leaves

        for node in nodes_to_label:
            theta = node.theta
            lx, ly = polar_to_cart(label_r, theta)
            node_name = getattr(node, "sci_name", node.name)

            # Rotation logic
            norm_theta = theta % 360

            if 90 < norm_theta < 270:
                # Left Side
                angle = 180 - norm_theta
                xanchor = "right"
            else:
                # Right Side
                angle = -norm_theta
                xanchor = "left"

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
            theta = node.theta
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

        # logic
        if triggered_id == "reset-btn":
            new_root_name = "Bacteria"
            new_root_rank = "superkingdom"

        elif triggered_id == "tree-graph" and click_data:
            # check what was clicked
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
