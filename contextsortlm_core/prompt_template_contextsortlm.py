CONTEXTSORTLM_SUMMARY_PROMPT = [
'''Construct a general instruction for how to arrange objects in a user\'s
home personalized to this user. Input and output must be in JSON format. The
input field "rules" contains a list of rules about where to place objects, generated
from user examples. The input field "receptacles" contains a list of receptacles in
the home. Output "summary" as a concise instruction about where to place object categories.
If multiple rules mention the same or similar objects, it is ok to combine them.
Do not combine rules for semantically different objects. For conflicting rules,
list all the different receptacles that the object can be placed on. Focus on household
objects, as the goal is to arrange a person\'s home.''',
'''{"rules":[
"Put cornmeal on the middle rack, coffee on the table, chips on the shelf, and dried fruits and cashews in the plastic box.",
"Put bread crumbs on the top rack, and almonds in the plastic box",
"Put vinegar in the middle rack, tea on the table, and the protein bar on the shelf",
"Put peanut oil in the bottom rack, and figs in the plastic box"
],
"receptacles":["top rack","middle rack","table","shelf","plastic box"]}''',
'''{"summary":
{
"dry ingredients":["top rack","middle rack"],
"liquid ingredients":["middle rack","bottom rack"],
"tea and coffee":["table"],
"packaged snacks":["shelf"],
"dried fruits and nuts":["plastic box"],
}
}'''
]


CONTEXTSORTLM_PLACEMENT_PROMPT = [
'''# Summary: {"clothes": ["laundry basket"], "toys": ["storage box"]}
objects = ["socks", "toy car", "shirt", "Lego brick"]
receptacles = ["laundry basket", "storage box"]
pick_and_place("socks", laundry basket")
pick_and_place("toy car",''',
'''"storage box")
pick_and_place("shirt", "laundry basket")
pick_and_place("Lego brick", "storage box")
''',
]
