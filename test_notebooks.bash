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

# File that stores the list of successfully tested notebook absolute paths.
TESTED_FILE="tested.txt"

# Create tested.txt if it does not exist.
if [ ! -f "$TESTED_FILE" ]; then
    touch "$TESTED_FILE"
fi


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

    # Execute the notebook using nbconvert.
    # If any cell fails, nbconvert returns a nonzero exit status.
    jupyter nbconvert --to notebook --execute "$notebook" --output "$tmp_output" 2>error.log
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "Error encountered while executing notebook: $notebook"
        echo "Error details:"
        cat error.log
        # Clean up temporary files and exit.
        rm error.log "$tmp_output"
        exit 1
    else
        echo "Notebook executed successfully: $notebook"
    fi
    rm error.log "$tmp_output"
}

# Find all .ipynb files recursively in the specified directory.
IFS=$'\n'
file_list=($(find "$SEARCH_DIR" -type f -name "*.ipynb"))
unset IFS

# Process each notebook found.
for notebook in "${file_list[@]}"; do
    # Get the absolute path of the notebook.
    abs_path=$(readlink -f "$notebook")

    # If the notebook is already logged in tested.txt, skip it.
    if grep -Fqx "$abs_path" "$TESTED_FILE"; then
        echo "Skipping already tested notebook: $abs_path"
        continue
    fi

    # Skip notebook if it matches any exclude pattern.
    if should_exclude "$abs_path"; then
        echo "Skipping excluded notebook: $abs_path"
        continue
    fi

    # Execute the notebook; if it executes successfully, log its absolute path.
    execute_notebook "$abs_path"
    echo "$abs_path" >> "$TESTED_FILE"
done

echo "All notebooks executed successfully."