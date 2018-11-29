import argparse
import os
import random
from utils.io import ensure_parent_exists

def split_train_val(in_dir, out_dir_train, out_dir_val, val_frac):
    """
    Splits data into training and validation.

    Args:
        in_dir : Path to input directory.
        out_dir_train : Path to output dir where training samples will be stored.
        out_dir_val : Path to output dir where validation samples will be stored.
        val_frac : Fraction of samples that should go to validation set.
    """
    for file_name in os.listdir(in_dir):
        in_file_path = os.path.join(in_dir, file_name)
        if os.path.isfile(in_file_path):
            with open(in_file_path, 'r', encoding='utf-8') as in_file:
                lines = in_file.readlines()
                random.shuffle(lines)
                train_size = int(len(lines) * (1.0 - val_frac))
                out_file_path = os.path.join(out_dir_train, file_name)
                ensure_parent_exists(out_file_path)
                with open(out_file_path, 'w', encoding='utf-8') as out_file:
                    for line in lines[:train_size]:
                        out_file.write(line)
                out_file_path = os.path.join(out_dir_val, file_name)
                ensure_parent_exists(out_file_path)
                with open(out_file_path, 'w', encoding='utf-8') as out_file:
                    for line in lines[train_size:]:
                        out_file.write(line)

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', help='Path to input dir.', required=True)
    parser.add_argument('-out_dir_train', help='Path to output dir where training samples will be stored.', required=True)
    parser.add_argument('-out_dir_val', help='Path to output dir where validation samples will be stored.', required=True)
    parser.add_argument('-val_frac', help='Fraction of samples that should go to validation set.', required=True)
    args = vars(parser.parse_args())
    # Collects text lines and places them to files in output directory.
    split_train_val(args['in_dir'], args['out_dir_train'], args['out_dir_val'], float(args['val_frac']))
