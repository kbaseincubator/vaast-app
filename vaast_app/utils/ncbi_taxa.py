"""
Process-safe loader for NCBITaxa class.

On first load to a new environment (such as a deployment, etc.), instances of ``ete3.NCBITaxa`` attempt to download the
NCBI taxonomy databases. This is unnecessary to do from multiple processes that reference the same data, and can also
lead to corrupted data if multiple processes write to the same file.

``ete3.NCBITaxa`` must be initialized using its constructor, but this will only require download steps the first time
that it is instantiated within a given container.
"""

from ete3 import NCBITaxa

from vaast_app.utils.concurrent import OperationLock


def get_ncbi_taxa() -> NCBITaxa:
    """
    Get an instance of NCBITaxa. Process-safe.

    :return: NCBITaxa instance
    """
    with OperationLock("ncbi-taxa"):
        return NCBITaxa()
