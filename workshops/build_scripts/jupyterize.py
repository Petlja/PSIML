 ## Converts pyhton script into a notebook.
 
# See [this post](https://stackoverflow.com/questions/23292242/converting-to-not-from-ipython-notebook-format) on StackOverflow.

import argparse
import os

argParser = argparse.ArgumentParser()
argParser.add_argument('-in', help='Path to input python file.', required=True)
argParser.add_argument('-out', help='Path to output jupyter notebook file.', required=True)

arguments = vars(argParser.parse_args())

inFile = arguments["in"]
outFile = arguments["out"]

from nbformat import v3, v4
with open(inFile) as fpin:
    text = fpin.read()
text += "# <markdowncell>\n # If you can read this, reads_py() is no longer broken!\n"
  
nbook = v3.reads_py(text)
nbook = v4.upgrade(nbook) # Upgrade v3 to v4
jsonform = v4.writes(nbook) + "\n"

directory = os.path.dirname(outFile)
os.makedirs(directory, exist_ok=True)
with open(outFile, "w") as fpout:
    fpout.write(jsonform)