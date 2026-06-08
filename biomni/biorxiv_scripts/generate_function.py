#!/usr/bin/env python3
"""Command-line tool to generate Python functions from task descriptions using the function_generator agent."""

import argparse
import os
import re
import sys
from pathlib import Path

# Add the parent directory to the path so we can import bioagentos
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from bioagentos.agent.function_generator import function_generator


def sanitize_filename(name):
    """Convert a string to a valid filename by replacing invalid characters.

    Args:
        name: The string to convert

    Returns:
        A sanitized filename

    """
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r"[^\w\s-]", "_", name.lower())
    sanitized = re.sub(r"[\s-]+", "_", sanitized)
    return sanitized


def main():
    """Main function for the command-line tool."""
    parser = argparse.ArgumentParser(description="Generate a Python function from a task description")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="Task description (if not provided, will prompt for input)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="generated_functions",
        help="Directory to save the generated function (default: generated_functions)",
    )
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        help="Base filename for the generated function (without extension, default: derived from function name)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="claude-3-7-sonnet-latest",
        help="LLM model to use (default: claude-3-7-sonnet-latest)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature setting for the LLM (default: 0.7)",
    )

    args = parser.parse_args()

    # If no task is provided, prompt for it
    if not args.task:
        print("Please enter your task description (type 'END' on a new line when finished):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        task_description = "\n".join(lines)
    else:
        task_description = args.task

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the function generator agent
    agent = function_generator(llm=args.model, temperature=args.temperature)

    # Generate the function
    print(f"\nGenerating function using {args.model}...")
    result = agent.generate_function(task_description)

    # Extract the function name from the generated code
    function_name = "generated_function"  # Default name
    function_code = result["function_code"]
    if function_code:
        match = re.search(r"def\s+(\w+)", function_code)
        if match:
            function_name = match.group(1)

    # Determine the filename
    base_filename = sanitize_filename(args.filename) if args.filename else sanitize_filename(function_name)

    # Save the function to a file
    function_file = f"{args.output_dir}/{base_filename}.py"
    test_file = f"{args.output_dir}/test_{base_filename}.py"

    with open(function_file, "w") as f:
        f.write(function_code)

    with open(test_file, "w") as f:
        f.write(result["test_case"])

    # Print the results
    print("\n" + "=" * 80)
    print("GENERATED FUNCTION:")
    print("=" * 80)
    print(function_code)

    print("\n" + "=" * 80)
    print("TEST CASE:")
    print("=" * 80)
    print(result["test_case"])

    print(f"\nFunction saved to {function_file}")
    print(f"Test case saved to {test_file}")


if __name__ == "__main__":
    main()
