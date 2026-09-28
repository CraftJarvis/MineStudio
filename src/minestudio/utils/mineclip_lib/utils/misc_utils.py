import fnmatch
from typing import List, Union, Callable
from typing_extensions import Literal


def _match_patterns_helper(element, patterns):
    for p in patterns:
        if callable(p) and p(element):
            return True
        if fnmatch.fnmatch(element, p):
            return True
    return False


def match_patterns(
    item: str,
    include: Union[str, List[str], Callable, List[Callable], None] = None,
    exclude: Union[str, List[str], Callable, List[Callable], None] = None,
    *,
    precedence: Literal["include", "exclude"] = "exclude",
):
    """
    Args:
        include: None to disable `include` filter and delegate to exclude
        precedence: "include" or "exclude"
    """
    assert precedence in ["include", "exclude"]
    if exclude is None:
        exclude = []
    if isinstance(exclude, (str, Callable)):
        exclude = [exclude]
    if isinstance(include, (str, Callable)):
        include = [include]
    if include is None:
        # exclude is the sole veto vote
        return not _match_patterns_helper(item, exclude)

    if precedence == "include":
        return _match_patterns_helper(item, include)
    else:
        if _match_patterns_helper(item, exclude):
            return False
        else:
            return _match_patterns_helper(item, include)


def filter_patterns(
    items: List[str],
    include: Union[str, List[str], Callable, List[Callable], None] = None,
    exclude: Union[str, List[str], Callable, List[Callable], None] = None,
    *,
    precedence: Literal["include", "exclude"] = "exclude",
    ordering: Literal["original", "include"] = "original",
):
    """
    Args:
        ordering: affects the order of items in the returned list. Does not affect the
            content of the returned list.
            - "original": keep the ordering of items in the input list
            - "include": order items by the order of include patterns
    """
    assert ordering in ["original", "include"]
    if include is None or isinstance(include, str) or ordering == "original":
        return [
            x
            for x in items
            if match_patterns(
                x, include=include, exclude=exclude, precedence=precedence
            )
        ]
    else:
        items = items.copy()
        ret = []
        for inc in include:
            for i, item in enumerate(items):
                if item is None:
                    continue
                if match_patterns(
                    item, include=inc, exclude=exclude, precedence=precedence
                ):
                    ret.append(item)
                    items[i] = None
        return ret


def getitem_nested(cfg, key: str):
    """
    Recursively get key, if key has '.' in it
    """
    keys = key.split(".")
    for k in keys:
        assert k in cfg, f'{k} in key "{key}" does not exist in config'
        cfg = cfg[k]
    return cfg


def setitem_nested(cfg, key: str, value):
    """
    Recursively get key, if key has '.' in it
    """
    keys = key.split(".")
    for k in keys[:-1]:
        assert k in cfg, f'{k} in key "{key}" does not exist in config'
        cfg = cfg[k]
    cfg[keys[-1]] = value


def getattr_nested(obj, key: str):
    """
    Recursively get attribute
    """
    keys = key.split(".")
    for k in keys:
        assert hasattr(obj, k), f'{k} in attribute "{key}" does not exist'
        obj = getattr(obj, k)
    return obj


def setattr_nested(obj, key: str, value):
    """
    Recursively set attribute
    """
    keys = key.split(".")
    for k in keys[:-1]:
        assert hasattr(obj, k), f'{k} in attribute "{key}" does not exist'
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)
