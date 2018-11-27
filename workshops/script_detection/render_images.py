import wikipedia
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

# List of languages given with language codes.
languages = ["ko", "el", "en", "sr"]

# Mapping from language code to font to be used for rendering.
font_by_language = {
    "ko" : "malgun.ttf",
    "en" : "arial.ttf",
    "sr" : "arial.ttf",
    "el" : "arial.ttf"
}

def render_sample(text, font_name, height):
    """
    Render image of the given text using the given font.

    Args:
        text :      Text to be rendered.
        font_name : Font to be used in rendering.

    Returns:
        Rendered image.
    """
    img = Image.new('RGB', (10, 10), color = 'white')
    img_draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(font_name, 40)
    text_size = img_draw.textsize(text, font=fnt)
    img = img.resize(text_size)
    img_draw = ImageDraw.Draw(img)
    img_draw.text((0, 0), text, font=fnt, fill='black')

    # Resize rendered image ot given height.
    scale = height / text_size[1]
    width = int(text_size[0] * scale)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    return img_resized

def create_images(in_dir, out_dir, height):
    """
    Based on the language specific files within given dir (files containing textual lines) renders corresponding
    images and saved them to given output directory.

    Args:
        in_dir - Path to input directory containing textual files.
        out_dir - Path to output directory where images will be saved.
    """
    files = os.listdir(in_dir)
    # Go over all files.
    for file in files:
        file_path = os.path.join(in_dir, file)
        # Determine language based on file name.
        language = os.path.splitext(file)[0]
        # Take appropriate font for language.
        font_name = font_by_language[language]
        # Create output dir (name it after language code).
        out_lang_dir = os.path.join(out_dir, language)
        os.makedirs(out_lang_dir)
        # Load all lines from the file.
        in_file = open(file_path, "r", encoding="utf-8")
        text_lines = in_file.read().split("\n")
        in_file.close()
        # Go over lines and render them.
        current_sample = 0
        for text_line in text_lines:
            if text_line == "":
                continue
            img = render_sample(text_line, font_name, height)
            img_name = str(current_sample) + ".png"
            img_path = os.path.join(out_lang_dir, img_name)
            # Save image to output language directory.
            img.save(img_path)
            current_sample += 1

def calculate_max_height(in_dir):
    """
    Goes through all lines and finds maximal rendered image height.

    Args:
        in_dir - Path to input directory containing textual files.
    """
    height = 0
    files = os.listdir(in_dir)
    # Go over all files.
    for file in files:
        file_path = os.path.join(in_dir, file)
        # Determine language based on file name.
        language = os.path.splitext(file)[0]
        # Take appropriate font for language.
        font_name = font_by_language[language]
        # Load all lines from the file.
        in_file = open(file_path, "r", encoding="utf-8")
        text_lines = in_file.read().split("\n")
        in_file.close()
        # Go over lines.
        for text_line in text_lines:
            if text_line == "":
                continue
            img = Image.new('RGB', (10, 10), color = 'white')
            img_draw = ImageDraw.Draw(img)
            fnt = ImageFont.truetype(font_name, 40)
            text_size = img_draw.textsize(text_line, font=fnt)
            if text_size[1] > height:
                height = text_size[1]

    return height

if __name__ == "__main__":
    # Parse arguments. 
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', help='Path to input dir where text files are be stored.', required=True)
    parser.add_argument('-out_dir', help='Path to output dir where images will be stored.', required=True)
    args = vars(parser.parse_args())
    # Calculate maximal image height.
    height = calculate_max_height(args["in_dir"])
    # Create images and place them in output directory.
    create_images(args["in_dir"], args["out_dir"], height)
