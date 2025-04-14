#!/bin/bash
# run_notebooks.sh
#
# This script finds and executes all .ipynb notebooks recursively in a given folder.
# It excludes notebooks whose file paths match any of the given exclusion patterns.
# Notebooks are run sequentially using 'jupyter nbconvert --execute'.
# If an error occurs during execution (such as an error in a notebook cell),
# the script stops and reports the notebook file that failed along with the error details.
#
# Usage:
#   ./run_notebooks.sh [search_directory]
#
# If no search_directory is provided, the current directory (.) is used.

# Set the directory in which to search; default to current directory.
SEARCH_DIR="${1:-.}"

# Define an array of patterns for files to be excluded.
# Adjust or add patterns as required.
EXCLUDES=(".ipynb_checkpoints" "another_pattern_to_exclude")

# Function: Check if a file path matches any exclude pattern.
should_exclude() {
    local file="$1"
    for pattern in "${EXCLUDES[@]}"; do
        if [[ "$file" == *"$pattern"* ]]; then
            return 0  # True – file should be excluded.
        fi
    done
    return 1  # False – file is not to be excluded.
}

# Function: Execute a given notebook and halt on error.
execute_notebook() {
    local notebook="$1"
    echo "Executing notebook: $notebook"

    # Create a temporary file to store the executed notebook.
    tmp_output=$(mktemp --suffix=.ipynb)

    # Execute the notebook; any error in a cell will cause a non-zero exit status.
    # The executed notebook is saved to the temporary file (and removed later).
    # Error output is captured in error.log.
    jupyter nbconvert --to notebook --execute "$notebook" --output "$tmp_output" 2>error.log
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "Error encountered while executing notebook: $notebook"
        echo "Error details:"
        cat error.log
        # Clean up temporary files and stop processing further notebooks.
        rm error.log "$tmp_output"
        exit 1
    else
        echo "Notebook executed successfully: $notebook"
    fi
    rm error.log "$tmp_output"
}

# Find all .ipynb files recursively in the specified directory.
# Using IFS workaround to handle filenames with spaces.
IFS=$'\n'
file_list=($(find "$SEARCH_DIR" -type f -name "*.ipynb"))
unset IFS

# Loop through each found notebook.
for notebook in "${file_list[@]}"; do
    # Skip the notebook if it matches any of the exclusion patterns.
    if should_exclude "$notebook"; then
        echo "Skipping excluded notebook: $notebook"
        continue
    fi

    # Execute the notebook and stop if an error occurs.
    execute_notebook "$notebook"
done

echo "All notebooks executed successfully."
