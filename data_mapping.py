# -*- coding: utf-8 -*-
"""data_mapping.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sstJhtXqi2x2fHiBP978OjokVpaISSIa
"""

import json

def map_data_to_objects(text_dict, summary_dict):
    mapped_data = {}

    for filename in text_dict.keys():
        mapped_data[filename] = {
            'text': text_dict[filename],
            'summary': summary_dict.get(filename, "No summary available")
        }

    return mapped_data

def save_mapping_to_file(mapped_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(mapped_data, f, indent=4)
    print(f"Mapping saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Example dictionaries from previous steps
    text_dict = {
        'object1.png': 'Extracted text from object 1',
        'object2.png': 'Extracted text from object 2',
        'object3.png': 'Extracted text from object 3'
    }

    summary_dict = {
        'object1.png': 'Summary for object 1.',
        'object2.png': 'Summary for object 2.',
        'object3.png': 'Summary for object 3.'
    }

    # Map data
    mapped_data = map_data_to_objects(text_dict, summary_dict)

    # Save the mapping to a JSON file
    output_file = 'mapped_data.json'
    save_mapping_to_file(mapped_data, output_file)

    # Print the mapped data
    print("Mapped data:", mapped_data)