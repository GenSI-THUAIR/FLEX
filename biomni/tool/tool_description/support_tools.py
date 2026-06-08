description = [
    {
        "description": "Executes the provided Python command in the notebook environment and returns the output.",
        "name": "run_python_repl",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Python command to execute in the notebook environment",
                "name": "command",
                "type": "str",
            }
        ],
    },
    {
        "description": "Read the source code of a function from any module path.",
        "name": "read_function_source_code",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Fully qualified function name "
                "(e.g., "
                "'bioagentos.tool.support_tools.write_python_code')",
                "name": "function_name",
                "type": "str",
            }
        ],
    },
]
