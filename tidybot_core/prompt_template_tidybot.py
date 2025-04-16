"""Tidybot prompt templates."""
import tiktoken

TIDYBOT_SUMMARY_TEMPLATE = '''objects = ["dried figs", "protein bar", "cornmeal", "Macadamia nuts", "vinegar", "herbal tea", "peanut oil", "chocolate bar", "bread crumbs", "Folgers instant coffee"]
receptacles = ["top rack", "middle rack", "table", "shelf", "plastic box"]
pick_and_place("dried figs", "plastic box")
pick_and_place("protein bar", "shelf")
pick_and_place("cornmeal", "top rack")
pick_and_place("Macadamia nuts", "plastic box")
pick_and_place("vinegar", "middle rack")
pick_and_place("herbal tea", "table")
pick_and_place("peanut oil", "middle rack")
pick_and_place("chocolate bar", "shelf")
pick_and_place("bread crumbs", "top rack")
pick_and_place("Folgers instant coffee", "table")
# Summary: Put dry ingredients on the top rack, liquid ingredients in the middle rack, tea and coffee on the table, packaged snacks on the shelf, and dried fruits and nuts in the plastic box.

objects = ["yoga pants", "wool sweater", "black jeans", "Nike shorts"]
receptacles = ["hamper", "bed"]
pick_and_place("yoga pants", "hamper")
pick_and_place("wool sweater", "bed")
pick_and_place("black jeans", "bed")
pick_and_place("Nike shorts", "hamper")
# Summary: Put athletic clothes in the hamper and other clothes on the bed.

objects = ["Nike sweatpants", "sweater", "cargo shorts", "iPhone", "dictionary", "tablet", "Under Armour t-shirt", "physics homework"]
receptacles = ["backpack", "closet", "desk", "nightstand"]
pick_and_place("Nike sweatpants", "backpack")
pick_and_place("sweater", "closet")
pick_and_place("cargo shorts", "closet")
pick_and_place("iPhone", "nightstand")
pick_and_place("dictionary", "desk")
pick_and_place("tablet", "nightstand")
pick_and_place("Under Armour t-shirt", "backpack")
pick_and_place("physics homework", "desk")
# Summary: Put workout clothes in the backpack, other clothes in the closet, books and homeworks on the desk, and electronics on the nightstand.'''

TIDYBOT_PLACEMENT_TEMPLATE = '''# Summary: Put clothes in the laundry basket and toys in the storage box.
objects = ["socks", "toy car", "shirt", "Lego brick"]
receptacles = ["laundry basket", "storage box"]
pick_and_place("socks", "laundry basket")
pick_and_place("toy car", "storage box")
pick_and_place("shirt", "laundry basket")
pick_and_place("Lego brick", "storage box")'''

if __name__ == "__main__":
    enc = tiktoken.encoding_for_model("gpt-4")
    for text in [TIDYBOT_PLACEMENT_TEMPLATE, TIDYBOT_SUMMARY_TEMPLATE]:
        tokens = enc.encode(text)
        print(len(tokens))
