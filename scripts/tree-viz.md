# Taxonomy Tree Visualization

## Overview

- The full taxonomy tree for the VAAST collection (named `full_collection_tree.nwk`) is very large and complex,
  and is not easily visualized.
- The app in this repository would benefit from a more compact visualization of the taxonomy tree with
  zoom features by taxonomic rank (e.g., zooming in to the bacterial phylum would show all bacterial taxa).

## Implementation goals

- Create a plotly-based visualization of the taxonomy tree with zoom features by taxonomic rank.
- The initial zoom would be at the phylum level and would consist of a radial tree layout.
  - Specifically, the default view will be at the phylum level, and the tree would render down 2 levels to the order level. 
  - Colored pie wedges would surround each of the members of the same phylum.
  - A single label would be displayed for each phylum.
- When the user clicks on a colored range, the visualization would zoom in to entries in that selection, and 
  would display a radial tree layout with the root at the selected node, render down to the subsequent 2 levels of the taxonomy.
  - For example, if a user clicks on the 'alphaproteobacteria' class range, the next view would show a radial 
    tree layout for all orders and families in the alpha proteobacteria class, rendered to display all orders and families in this class.
