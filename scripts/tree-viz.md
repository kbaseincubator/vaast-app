# Taxonomy Tree Visualization

## Overview

A Plotly-based interactive visualization of the VAAST collection taxonomy tree. This tool facilitates exploration of large phylogenetic trees by implementing a "radial dendrogram" layout with zoom capabilities.

## Features

### 1. Radial Layout with Orthogonal Edges
- **Structure**: The tree is rendered as a radial dendrogram.
- **Edges**: Instead of diagonal or curved lines, edges are rendered orthogonally ("fork-like").
  - **Radial Segments**: Correspond to the branch length (evolutionary distance).
  - **Arc Segments**: Connect sibling branches at the parent's radius.
- **Sectors**: High-level clades (e.g., Phyla) are highlighted with colored background wedges (sectors) that encompass all their descendants.

### 2. Intelligent Label Placement
- **Positioning**: Labels are placed tangentially along the perimeter of the tree.
- **Centering**: The angular position of a label is calculated as the **geometric center** of the arc it represents (`(start_angle + end_angle) / 2`). This ensures labels for large groups (like *Pseudomonadota*) are visually balanced over their sector.
- **Radial Offset**: Labels are pushed outward based on the length of their text (`label_r + 0.02 * len(text)`) to prevent overlapping with the tree tips or other truncated labels.
- **Rotation & Readability**:
  - Labels are rotated to be perpendicular to the radius (tangential).
  - **Readability Flip**: To ensure text is never upside-down, the rotation logic flips the text by 180° at specific angles.
    - **Right Hemisphere (-90° to 90°)**: Text reads "upwards" (counter-clockwise).
    - **Left Hemisphere (90° to 270°)**: Text reads "downwards" (clockwise).
    - **Bottom Quadrant**: Special logic ensures labels in the bottom-right (270°-360°) are flipped correctly to be readable relative to the bottom-left labels.

### 3. Interactive Zoom & Navigation
- **Initial View**: Defaults to the *Bacteria* superkingdom, coloring by Phylum and showing details down to the Order level.
- **Click-to-Zoom**: Clicking on a sector or node re-roots the visualization to that taxon.
  - The view updates to show the selected taxon as the center.
  - The "Color Rank" and "Leaf Rank" automatically adjust (e.g., clicking a Phylum colors by Class and shows Families).
- **Reset**: A "Reset View" button returns the visualization to the top-level Bacteria view.
- **Breadcrumbs**: Displays the current root and rank path.

## Technical Implementation

- **Library**: Python `Plotly Graph Objects` for rendering, `Dash` for interactivity.
- **Tree Processing**: `ete3` toolkit for Newick parsing and tree traversal.
- **Coordinate System**:
  - Computations are performed in **Polar Coordinates** (`r`, `theta`).
  - Converted to **Cartesian Coordinates** (`x`, `y`) for Plotly rendering.
- **Algorithm**:
  1. **Load & Collapse**: The full tree is loaded and "collapsed" at a target rank (e.g., Order) to reduce visual clutter. Descendant counts are aggregated.
  2. **Layout Calculation**:
     - Leaves are assigned uniform angular widths.
     - Internal nodes derive their angles from the average of their children.
     - Sectors derive their start/end angles from the minimum/maximum angles of their descendants.
  3. **Rendering**:
     - **Sectors**: Drawn as filled shapes using interpolated arcs.
     - **Edges**: Drawn as lines connecting nodes via orthogonal paths.
     - **Labels**: Placed and rotated using the custom logic described above.

## Usage

Run the development server:
```bash
python scripts/develop_tree_explorer.py
```
This launches a Dash app where you can explore the visualization.
