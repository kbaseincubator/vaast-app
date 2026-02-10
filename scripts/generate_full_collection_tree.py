from pathlib import Path

import polars as pl

from vaast_app.utils.bacdive_utils import BacdiveAPISearcher, entries_in_vaast
from vaast_app.utils.ncbi_taxa import get_ncbi_taxa
from vaast_app.utils.tree_utils import TreeUtils


def _main() -> None:
    vaast_labels, vaast_ids = entries_in_vaast()
    ncbi_client = get_ncbi_taxa()

    # Remove any ids that are eukaryotic in origin
    vaast_ids = {
        taxid
        for taxid in vaast_ids
        if (lineage := ncbi_client.get_lineage(taxid)) is not None
        and 2759 not in lineage  # Not eukaryotic
        and 10239 not in lineage  # Not viral
    }
    bd_searcher = BacdiveAPISearcher.get_searcher()
    vaast_ids.update(
        bd_searcher.summary_df.select("species-taxid")
        .filter(pl.col("species-taxid").is_not_null())
        .collect()["species-taxid"]
    )
    tree = ncbi_client.get_topology(vaast_ids, intermediate_nodes=True)
    results = TreeUtils.SearchResult(tree, [], bd_searcher, []).to_data()["tree"]["newick"]
    results = results.replace(" (in: a-proteobacteria)", "")

    with open(Path(__file__).parent / "full_collection_tree.nwk", "w") as f:
        f.write(results)


if __name__ == "__main__":
    _main()
