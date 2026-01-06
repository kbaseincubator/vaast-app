"""
Compression algorithm to reduce size of BacDive database

Referenced (copied) from https://medium.com/@busybus/zipjson-3ed15f8ea85d
"""

import base64
import json
import zlib
from typing import Any, NewType, TypeVar

RawData = TypeVar("RawData")
JSONKeyType = NewType("JSONKeyType", str)

_ZIPJSON_KEY = JSONKeyType("base64(zip(o))")


def json_zip(j: RawData) -> dict[JSONKeyType, str]:
    """
    Convert dictionary to zipped JSON format

    The resulting data will be in the format: ``{_ZIPJSON_KEY: zipstring}`` and can be written to file.
    This can be decompressed using the ``json_unzip`` function in this module.

    :param j: Dictionary of data that supports ``json.dump``
    :return: Compressed data
    """
    _j = {
        # Encode compressed JSON data to base64 and then to ascii
        _ZIPJSON_KEY: base64.b64encode(
            # Compress stringified JSON data
            zlib.compress(json.dumps(j).encode("utf-8"))
        ).decode("ascii")
    }

    return _j


def json_unzip(j: dict[JSONKeyType, str]) -> Any:
    """
    Decompress dictionary that was compressed by the ``json_zip`` function in this module.

    :param j: Data that was returned by the ``json_zip`` function in this module
    :return: Decompressed data
    """
    try:
        # Validate expected data format
        assert set(j.keys()) == {_ZIPJSON_KEY}
    except AssertionError as err:
        raise RuntimeError("JSON not in the expected format {" + str(_ZIPJSON_KEY) + ": zipstring}") from err

    try:
        # Decode and decompress data string to JSON string
        _j1 = zlib.decompress(base64.b64decode(j[_ZIPJSON_KEY]))
    except zlib.error as err:
        raise RuntimeError("Could not decode/unzip the contents") from err

    try:
        # Convert JSON string to Python data structure
        _j2 = json.loads(_j1)
    except json.JSONDecodeError as err:
        raise RuntimeError("Could interpret the unzipped contents") from err

    return _j2
