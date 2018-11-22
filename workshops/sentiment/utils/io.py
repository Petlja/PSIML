import os
import json


def ensure_parent_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_to_json(data, output_file, sort_keys=False, indent=None):
    ensure_parent_exists(output_file)
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, sort_keys=sort_keys, indent=indent)


def load_from_json(input_file):
    with open(input_file) as json_data:
        return json.load(json_data)
