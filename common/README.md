# Common package

This package contains shared code and utilities for all subtasks. It provides the following submodules:

- `providers/`: Contains code for interacting with different LLM providers (local/cloud).

This package is designed to be imported by the subtask-specific code, allowing for code reuse and modularity across the different subtasks.

## Installation

In a new subtask without this package, you can install it using uv with the following command:

```bash
uv add ../../common # (adjust the path as needed based on your directory structure)
```

## Usage

Once installed, you can import the necessary modules from the `common` package in your subtask code. For example:

```python
from common.providers import LocalProvider, CloudProvider

provider = LocalProvider(model_name="qwen/qwen3-8b")
```
