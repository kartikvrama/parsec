from utils import constants
LOCATIONS_PH = "DXTHBGYR" # Placeholder to fill in environment placement locations.

APRICOT_PREFERENCE_PROMPT = '''--------- System Message ---------
You are an assistant who sees someone demonstrating how the environment is organized and summarizes that person's preference.  
--------- Instruction ---------
# Input  
You are given 2 demonstrations that show the before and after when a set of objects gets put into the environment. For each demonstration:  
- **"Final state of the environment"** describes what the environment looks like after the demonstration.

# Goal
Your goal is to generate one preference that is consistent with the demonstrations and explain what the user wants.  

# Instructions and Useful Information  
## Specific Locations in the Environment  
The environment has the following locations: DXTHBGYR
## Details About the Preferences That You Need to Output  
A preference is a short paragraph that specifies requirements for each category of items. There must be at least one requirement for each category. The type of requirement for each category can be different.

The requirement needs to be one of the following:  
- **Type-1. General Locations.**  
  The options are: DXTHBGYR 
- **Type-2. Relative Positions.**  
  The options are:  
  - "<category> must be placed together next to existing <category> regardless of which shelf they are on."  
  - "<category> must be placed on the same shelf next to <another category of objects>, and which specific shelf does not matter."  

In addition to giving specific requirements for each category of items, sometimes you may choose to add additional requirements. The options are:  
- **Type-3. Exception for Attribute**  
  - "<category> needs to be placed at <specific location 1>, but <attribute of category> needs to be placed at <specific location 2>."  
  - An attribute includes a subcategory of the object, the size/weight of the object, a specific feature of the object, etc.  
- **Type-4. Conditional on Space**  
  - "If there are less than <N> objects at <primary specific location>, I want <category> to be placed at <primary specific location>. Else, I want <category> to be placed at <second choice specific location>."  

Now, you need to generate **one preference** as a JSON object:  
- Each preference must be in natural language.  
- You must refer to the demonstrations and output the preferences in this format:  

```json
{
  "reasoning": "<You must explain your thought process behind generating this preference.>",
  "preference": "<You must write the preference in natural language here>"
}
```
'''

APRICOT_PLANNING_PROMPT = '''--------- System Message --------
You are an assistant that comes up with a plan for putting items into an environment given a list of items and a human's preference.
--------- Instruction --------
You must analyze a human's preferences and then come up with a plan to put items into an environment.

You will receive the following as input:
Optional[Feedback]: ...
Objects: ...
Locations: ...
Initial State: ...
Preference: ...

where:
- "Feedback" appears if the previous plan was geometrically infeasible.
  - It will contain the items that did not fit in the previous plan.
  - It is possible that the item could fit but the pick_and_place function call was incorrect (e.g. misspelled item or location).
- "Objects" is a list of items that need to be placed.
- "Locations" is a list of locations in the environment.
- "Initial State" is a dictionary whose keys are the locations in the environment and the values are a list of items currently in that location.
- "Preference" is a description of the human's preferences for where they like things in a environment.
- The preference should always be satisfied, even when attempting to place items that did not fit in the previous plan.

You must respond in the following format:

# Reflect: ...
# Reasoning: ...
pick_and_place(item1, location1)
pick_and_place(item2, location2)
...

where:
- "Reflect" should contain reasoning about the previous plan if it was geometrically infeasible.
  - Reflect on what went wrong and how you plan to fix it.
  - The plan must abide by the human's preference.
- "Reasoning" should contain the reasoning for your plan.
  - Reason about how best to place the items in the environment based on the human's preference.
  - If the reflection involves repositioning items, ensure that the human's preference is still satisfied.
- "pick_and_place(item, location)" is a function call that places the item in the location in the environment.
  - This is your plan of action.

Each time you are prompted to generate a new plan, the environment is reset to its initial state. Use this to your advantage to come up with a better plan. It is absolutely necessary to satisfy the human's preference; geometric infeasibility only suggests that objects should be placed in different locations that still satisfy the preference.

--------- Example Input --------
`Objects: ["milk", "cheese", "apple", "orange"]
Locations: ["top shelf", "left side of top shelf", "right side of top shelf", "middle shelf", "left side of middle shelf", "right side of middle shelf", "bottom shelf", "left side of bottom shelf", "right side of bottom shelf"]
Initial State: {}
Preference: "I like putting dairy on the top shelf and fruits on the right side of middle shelf."
--------- Example Response --------
# Reasoning: Milk and cheese are dairy products, which based on the preference need to be on the top shelf. Apples and oranges are fruits, which need to be on the right side of the middle shelf.
pick_and_place("milk", "top shelf")
pick_and_place("cheese", "top shelf")
pick_and_place("apple", "right side of middle shelf")
pick_and_place("orange", "right side of middle shelf")
--------- Example Input --------
Objects: ["milk", "cheese", "apple", "orange"]
Locations: ["top shelf", "left side of top shelf", "right side of top shelf", "middle shelf", "left side of middle shelf", "right side of middle shelf", "bottom shelf", "left side of bottom shelf", "right side of bottom shelf"]
Initial State: {"left side of middle shelf": ["peach", "cherries"]}
Preference: "Fruits must be placed next to other existing fruits regardless of which shelf they are on. Dairy products must be on the right side of the bottom shelf."
--------- Example Response --------
# Reasoning: Apples and oranges are fruits, which need to be placed next to other existing fruits. 
Existing fruits in the fridge are "peach" and "cherries", which are at the left side of the middle shelf, so apple and orange should be placed there as well. Milk and cheese are dairy products, which need to be placed on the right side of the bottom shelf.
pick_and_place("apple", "left side of middle shelf")
pick_and_place("orange", "left side of middle shelf")
pick_and_place("milk", "right side of bottom shelf")
pick_and_place("cheese", "right side of bottom shelf")
--------- Example Input --------
Objects: ["milk", "cheese", "apple", "orange"]
Locations: ["top shelf", "left side of top shelf", "right side of top shelf", "middle shelf", "left side of middle shelf", "right side of middle shelf", "bottom shelf", "left side of bottom shelf", "right side of bottom shelf"]
Initial State: {"right side of top shelf": ["cabbage", "corn"]}
Preference: "Fruits must be placed next to other exisiting vegetables regardless of which shelf they are on. Dairy products on the be the right side of bottom shelf. "
--------- Example Response --------
# Reasoning: Apples and oranges are fruits, which needs to be placed next to other exisiting vegetables. Existing vegetables in the fridge are "cabbage" and "corn", which are at the right side of top shelf, so apple and oranges should be placed there as well. Milk and cheese are diary product, which needs to be placed on the right side of bottom shelf.
pick_and_place("apple", "right side of top shelf")
pick_and_place("orange","right side of top shelf")
pick_and_place("milk", "right side of bottom shelf")
pick_and_place("cheese", "right side of bottom shelf")
--------- Example Input --------
Objects: ["milk", "cheese", "apple", "orange"]
Locations: ["top shelf", "left side of top shelf", "right side of top shelf", "middle shelf", "left side of middle shelf", "right side of middle shelf", "bottom shelf", "left side of bottom shelf", "right side of bottom shelf"]
Initial State: {"left side of middle shelf": ["peach", "cherries"]}
Preference: "Most dairy product should be placed on the left side of top shelf, but cheese product should be placed on the right side of middle shelf. Fruits should be placed on the left side of middle shelf."
--------- Example Response --------
# Reasoning: Cheese is a cheese product, so it needs to be placed at right side of middle shelf. Milk is a dairy product, so it should be placed on the left side of top shelf. Apples and oranges are fruits, which needs to be on the left side of middle shelf.
pick_and_place("cheese", "right side of middle shelf")
pick_and_place("milk", "left side of top shelf")
pick_and_place("apple", "left side of middle shelf")
pick_and_place("orange","left side of middle shelf")
--------- Example Input --------
Objects: ["milk", "cheese", "apple", "orange"]
Locations: ["top shelf", "left side of top shelf", "right side of top shelf", "middle shelf", "left side of middle shelf", "right side of middle shelf", "bottom shelf", "left side of bottom shelf", "right side of bottom shelf"]
Initial State: {"right side of bottom shelf": ["yogurt", "butter"]}
Preference: "If the right side of bottom shelf has less than 3 items, dairy products can be placed there. Else, you must place them at the left side of top shelf. Fruits should be placed on the left side of middle shelf."
--------- Example Response --------
# Reasoning: Right side of bottom shelf can fit one more dairy product, so I will put milk there. Since now there are 3 items at the right side of bottom shelf, I must put cheese on the left side of top shelf . Apples and oranges are fruits, which needs to be on the left side of middle shelf.
pick_and_place("milk", "right side of bottom shelf")
pick_and_place("cheese", "left side of top shelf")
pick_and_place("apple", "left side of middle shelf")
pick_and_place("orange","left side of middle shelf")
--------- Example Input --------
Objects: ["milk", "cheese", "apple", "melon"]
Locations: ["top shelf", "middle shelf", "bottom shelf"]
Initial State: {
"top shelf": ["yogurt", "butter"],
"middle shelf": ["watermelon", "pizza box"]
}
Preference: "I don"t want any of the fridge shelves to be too crowded."
--------- Example Response --------
pick_and_place("milk", "top shelf")
pick_and_place("cheese", "top shelf")
pick_and_place("apple", "middle shelf")
pick_and_place("melon", "middle shelf")
--------- Example Input --------
Feedback: The previous plan was geometrically infeasible. The items that did not fit were ["melon", "apple"].
Objects: ["milk", "cheese", "apple", "melon"]
Locations: ["top shelf", "middle shelf", "bottom shelf"]
Initial State: {
"top shelf": ["yogurt", "butter"],
"middle shelf": ["watermelon", "pizza box"]
}
Preference: "I don"t want any of the fridge shelves to be too crowded."
--------- Example Response --------
# Reflect: The melon and apple did not fit in the previous plan. This must mean that the middle shelf is too crowded.
# Reasoning: Instead of placing the apple and melon on the middle shelf, they can be placed on the bottom shelf which is empty.
pick_and_place("milk", "top shelf")
pick_and_place("cheese", "top shelf")
pick_and_place("apple", "bottom shelf")
pick_and_place("melon", "bottom shelf")
'''
