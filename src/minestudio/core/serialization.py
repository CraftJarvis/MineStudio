"""Portable, non-pickle encoding for info values, including typed NumPy arrays.

Mappings use explicit entries so integer keys and reserved-looking user keys
round-trip. Manifests remain ordinary JSON; only arbitrary info uses this codec.
"""

import base64
import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np


def encode_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Info floats must be finite")
        return value
    if isinstance(value, np.generic):
        return encode_value(value.item())
    if isinstance(value, np.ndarray):
        array = cast("np.ndarray[Any, Any]", value)
        if array.dtype.kind not in "biuf" or array.dtype.hasobject:
            raise TypeError("Only numeric/bool NumPy arrays can be recorded")
        return {
            "type": "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if any(not isinstance(k, (str, int)) or isinstance(k, bool) for k in mapping):
            raise TypeError("Info mapping keys must be strings or integers")
        return {
            "type": "mapping",
            "items": [[key, encode_value(item)] for key, item in mapping.items()],
        }
    if isinstance(value, (tuple, list)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [
                encode_value(item) for item in cast(tuple[object, ...] | list[object], value)
            ],
        }
    raise TypeError(f"Unsupported info value: {type(value).__name__}")


def decode_value(value: Any) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Invalid non-finite info value")
        return value
    if not isinstance(value, dict):
        raise ValueError("Malformed info encoding")
    node = cast(dict[str, Any], value)
    if node.get("type") == "ndarray" and set(node) == {"type", "dtype", "shape", "data"}:
        dtype = np.dtype(node["dtype"])
        shape = node["shape"]
        if (
            dtype.kind not in "biuf"
            or dtype.hasobject
            or not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in cast(list[object], shape))
        ):
            raise ValueError("Invalid array dtype/shape")
        shape = cast(list[int], shape)
        raw = base64.b64decode(node["data"], validate=True)
        if math.prod(shape) * dtype.itemsize != len(raw):
            raise ValueError("Array shape does not match its data")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if set(node) != {"type", "items"} or not isinstance(node["items"], list):
        raise ValueError("Malformed info container")
    if node["type"] == "mapping":
        result: dict[str | int, object] = {}
        for entry in cast(list[Any], node["items"]):
            if not isinstance(entry, list):
                raise ValueError("Invalid info mapping entry")
            pair = cast(list[Any], entry)
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], (str, int))
                or isinstance(pair[0], bool)
                or pair[0] in result
            ):
                raise ValueError("Invalid or duplicate info mapping key")
            result[pair[0]] = decode_value(pair[1])
        return result
    if node["type"] in ("tuple", "list"):
        items = [decode_value(item) for item in cast(list[Any], node["items"])]
        return tuple(items) if node["type"] == "tuple" else items
    raise ValueError("Unknown info encoding type")
