# MCP GraphQL

An MCP (Model Control Protocol) server that enables interaction with GraphQL APIs.

## Description

MCP GraphQL is a tool that implements the Model Control Protocol (MCP) to provide a standardized interface for interacting with GraphQL APIs. It automatically exposes each GraphQL query as a separate MCP tool, allowing MCP-compatible clients to seamlessly communicate with GraphQL services.

## Features

- Each GraphQL query is exposed as a distinct MCP tool
- Tool parameters automatically match the corresponding GraphQL query parameters
- JSON schema for tool inputs is dynamically generated from GraphQL query parameters
- No schema definition required - simply provide the API URL and credentials
- Currently supports GraphQL queries (mutations support planned for future releases)
- Configurable authentication (Bearer, Basic, custom headers)
- Automatic handling of complex GraphQL types

## Requirements

- Python 3.11 or higher

## Installation

```bash
# Using pip
pip install mcp_graphql

# Or installation from source code
git clone https://github.com/your-username/mcp_graphql.git
cd mcp_graphql
pip install .
```

## Usage

### As a command line tool

```bash
mcp-graphql --api-url="https://api.example.com/graphql" --auth-token="your-token"
```

### Available options

- `--api-url`: GraphQL API URL (required)
- `--auth-token`: Authentication token (optional)
- `--auth-type`: Authentication type, default is "Bearer" (optional)
- `--auth-headers`: Custom authentication headers in JSON format (optional)

Example with custom headers:

```bash
mcp-graphql --api-url="https://api.example.com/graphql" --auth-headers='{"Authorization": "Bearer token", "X-API-Key": "key"}'
```

### As a library

```python
import asyncio
from mcp_graphql import serve

auth_headers = {"Authorization": "Bearer your-token"}
api_url = "https://api.example.com/graphql"

asyncio.run(serve(api_url, auth_headers))
```

## How It Works

MCP GraphQL automatically:

1. Introspects the provided GraphQL API
2. Creates an MCP tool for each available GraphQL query
3. Generates JSON schema for tool inputs based on query parameters
4. Handles type conversions between GraphQL and JSON

When a tool is called, the server:
1. Converts the tool call parameters to a GraphQL query
2. Executes the query against the API
3. Returns the results to the MCP client

## Planned Features

- Support for GraphQL mutations (with appropriate safeguards)
- Improved error handling and validation
- Pagination support for large result sets

## Development

### Setting up the development environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

### Linting

```bash
ruff check .
```

## License

[Include license information here]

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request or open an Issue.
