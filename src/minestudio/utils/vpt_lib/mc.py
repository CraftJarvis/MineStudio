import os
import json

mc_constants_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mc_constants.1.16.json"
)
all_data = json.load(open(mc_constants_file))

ALL_ITEMS = [item["type"] for item in all_data["items"]]

MINERL_ITEM_MAP = sorted(["none"] + ALL_ITEMS)