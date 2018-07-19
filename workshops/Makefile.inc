# Parent directory of this file is root directory of workshops.
ROOT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

# Build output directory.
BUILD_DIR = $(ROOT_DIR)/build/$(WORKSHOP)

# Path to the build script.
BUILD_SCRIPT = $(ROOT_DIR)/build_scripts/jupyterize.py

all:
	# Ensure output directory exist.
	mkdir -p $(BUILD_DIR)
	# Build jupyter notebooks using python build script.
	$(foreach NOTEBOOK_SOURCE_FILE, $(NOTEBOOK_SOURCE_FILES), \
		python $(BUILD_SCRIPT) -in $(ROOT_DIR)/$(WORKSHOP)/$(NOTEBOOK_SOURCE_FILE) -out $(BUILD_DIR)/$(NOTEBOOK_SOURCE_FILE).ipynb ; \
	)
	# Copy other python files.
	$(foreach OTHER_SOURCE_FILE, $(OTHER_SOURCE_FILES), \
		cp $(ROOT_DIR)/$(WORKSHOP)/$(OTHER_SOURCE_FILE) $(BUILD_DIR) ; \
	)
	# Copy resource directories.
	$(foreach RESOURCE_DIRECTORY, $(RESOURCE_DIRECTORIES), \
		cp -r $(ROOT_DIR)/$(WORKSHOP)/$(RESOURCE_DIRECTORY) $(BUILD_DIR) ; \
	)
