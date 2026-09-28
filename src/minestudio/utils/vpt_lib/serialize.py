import numpy as np
from typing import Dict, List, Tuple


def serialize_object(object):
    if isinstance(object, Dict):
        return {
            key: serialize_object(val) for key, val in object.items()
        }
    elif isinstance(object, List):
        return [
            serialize_object(val) for val in object
        ]
    elif isinstance(object, Tuple):
        return tuple(
            serialize_object(val) for val in object
        )
    elif isinstance(object, np.ndarray):
        # print(object)
        return ("numpy", object.tolist(), str(object.dtype), object.shape)
    else:
        return object