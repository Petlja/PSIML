import wikipedia
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

# List of languages given with language codes.
languages = ["ko", "el", "en", "sr"]

# List of search terms to be used to obtain text from Wikipedia.
terms_by_language = {
    "ko" : [ "아테네", "테살로니키"],
    "en" : ["london", "paris"],
    "sr" : ["london", "pariz"],
    "el" : [ "Λονδίνο", "Παρίσι"]
}

# Specifies how many words make up one example to be rendered.
words_per_sample = 10

def collect_text_lines(out_dir):
    """
    Collects text in different scripts. Collected text is obtained from Wikipedia. Each language text is palced in
    different file in output directory (file is named according to language code).

    Args:
        out_dir : Output directory path where collected text files are to be placed.
    """
    # Go over all languages
    for language in languages:
        # Set current language.
        wikipedia.set_lang(language)
        search_terms = terms_by_language[language]
        # Open output language file.
        out_lang_file_path = os.path.join(out_dir, language + ".txt")
        out_lang_file = open(out_lang_file_path, "w", encoding="utf-8")
        # Go over all search terms.
        for term in search_terms:
            # Get wiki page summary and split it in words.
            wiki_page = wikipedia.page(term)
            text = wiki_page.summary
            words = text.split()
            # Make text string as composition of certain number of words.
            curr_words = 0
            text_line = ""
            for w in range(len(words)):
                text_line += words[w]
                text_line += " "
                curr_words += 1
                if curr_words == words_per_sample:
                    # We have one line of text, save it to file.
                    out_lang_file.write(text_line)
                    out_lang_file.write("\n")
                    # Reset text line related variables.
                    curr_words = 0
                    text_line = ""

        out_lang_file.close()

if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir', help='Path to output dir where samples will be stored.', required=True)
    args = parser.parse_args()
    # Collects text lines and places them to files in output directory.
    collect_text_lines(vars(args)["out_dir"])
