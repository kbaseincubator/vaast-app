# Taxonomy Tree Visualization

## Overview

- The full taxonomy tree for the VAAST collection (named `full_collection_tree.nwk`) is very large and complex,
  and is not easily visualized.
- The app in this repository would benefit from a more compact visualization of the taxonomy tree with
  zoom features by taxonomic rank (e.g., zooming in to the bacterial phylum would show all bacterial taxa).

## Implementation goals

- Create a plotly-based visualization of the taxonomy tree with zoom features by taxonomic rank.
- The initial zoom would be at the class level and would consist of a radial tree layout with leaves in each 
  class collapsed into a single range for each class with a simple color and label.
- When the user clicks on a colored range, the visualization would zoom in to entries in that selection, and 
  would display a radial tree layout with leaves in each order collapsed into a single with a simple color and 
  label, focused on the next rank.
  - For example, if a user clicks on the 'alphaproteobacteria' class range, the next view would show a radial 
    tree layout for all orders in the alpha proteobacteria class, collapsed into a single range for each order.
