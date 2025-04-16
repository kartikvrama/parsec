import ast
import re


def fix_syntax_errors(input_string):
    # Fix common syntax errors
    # 1. Fix unquoted keys or values (e.g., same location -> 'same location')
    input_string = re.sub(
        r'(?<![\':\[\],])\b([a-zA-Z_ ]+)\b(?![\':\[\],])',
        r"'\1'",
        input_string
    )

    # 2. Fix missing commas between items (e.g., {'key1':'value1' 'key2':'value2'} -> {'key1':'value1', 'key2':'value2'})
    input_string = re.sub(r"(?<=[\]}\"'])\s*(?=[\[{])", ", ", input_string)

    # 3. Fix unbalanced brackets or braces (not fully comprehensive but helps)
    if input_string.count("[") != input_string.count("]"):
        input_string = input_string.replace("[", "]").replace("]", "[", 1)
    if input_string.count("{") != input_string.count("}"):
        input_string = input_string.replace("{", "}").replace("}", "{", 1)

    return input_string


def remove_duplicate_keys(dict_data):
    # Assuming duplicate keys should be overwritten by the last occurrence
    clean_dict = {}
    for key, value in dict_data.items():
        if isinstance(value, dict):
            clean_dict[key] = remove_duplicate_keys(value)
        else:
            clean_dict[key] = value
    return clean_dict


def string_to_dict(input_string):
    # Step 1: Fix syntax errors
    fixed_string = fix_syntax_errors(input_string)
    print(fixed_string)

    # Step 2: Convert the string to a dictionary
    try:
        dict_data = ast.literal_eval(fixed_string)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Failed to convert string to dictionary: {e}")
    # Step 3: Remove duplicate keys
    final_dict = remove_duplicate_keys(dict_data)
    return final_dict
