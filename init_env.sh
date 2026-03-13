#!/bin/bash
# Bash script to create and activate a Python virtual environment
# Usage: source ./init_env.sh [env_name]

env_name=${1:-venv}

if [ -d "$env_name" ]; then
    echo "Virtual environment '$env_name' already exists."
else
    echo "Creating virtual environment '$env_name'..."
    python -m venv "$env_name"
fi

echo "To activate the environment run:\n    source $env_name/bin/activate\n"
echo "Then install dependencies with:\n    pip install -r requirements.txt\n"