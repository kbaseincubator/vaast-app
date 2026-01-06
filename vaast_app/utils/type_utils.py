"""
NewType class definitions for identifying various database- and designation-level strings and integers
"""
from typing import NewType

BacdiveID = NewType("BacdiveID", int)
BacdiveRef = NewType("BacdiveRef", int)
BacdiveEntryRefID = NewType("BacdiveEntryRefID", int)
Feature = NewType("Feature", str)
FeatureType = NewType("FeatureType", str)
HeatmapID = NewType("HeatmapID", str)
TaxID = NewType("TaxID", int)
TaxName = NewType("TaxName", str)
