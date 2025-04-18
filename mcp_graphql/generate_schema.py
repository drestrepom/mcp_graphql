#!/usr/bin/env python
"""Generate GraphQL schema by executing introspection query against a remote API."""

import argparse
import asyncio
import json
import sys
from typing import Any

import httpx

# Default configuration - can be overridden with command line arguments
DEFAULT_ENDPOINT = "https://your-graphql-api-endpoint/graphql"
DEFAULT_OUTPUT = "schema.json"
DEFAULT_MAX_DEPTH = 7  # Increased default to match official tool


# Generate the official introspection query that matches get-graphql-schema tool
def generate_official_introspection_query(max_depth: int = 7) -> str:
    """Generate an introspection query that matches the official get-graphql-schema tool.

    Args:
        max_depth: Maximum nesting depth for type references (1-9)

    Returns:
        GraphQL introspection query

    """
    # Ensure max_depth is within reasonable bounds
    max_depth = max(1, min(9, max_depth))

    # Build the TypeRef fragment with the correct nesting depth
    type_ref_lines = ["fragment TypeRef on __Type {", "  kind", "  name"]

    current_indent = "  "
    for _i in range(max_depth):
        current_indent += "  "
        type_ref_lines.append(f"{current_indent[:-2]}ofType {{")
        type_ref_lines.append(f"{current_indent}kind")
        type_ref_lines.append(f"{current_indent}name")

    # Close brackets
    for _i in range(max_depth):
        current_indent = current_indent[:-2]
        type_ref_lines.append(f"{current_indent}}}")

    type_ref_lines.append("}")
    type_ref_fragment = "\n".join(type_ref_lines)

    # Main query that matches the official get-graphql-schema format
    return (
        """
query IntrospectionQuery {
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      ...FullType
    }
    directives {
      name
      description
      locations
      args {
        ...InputValue
      }
    }
  }
}

fragment FullType on __Type {
  kind
  name
  description
  fields(includeDeprecated: true) {
    name
    description
    args {
      ...InputValue
    }
    type {
      ...TypeRef
    }
    isDeprecated
    deprecationReason
  }
  inputFields {
    ...InputValue
  }
  interfaces {
    ...TypeRef
  }
  enumValues(includeDeprecated: true) {
    name
    description
    isDeprecated
    deprecationReason
  }
  possibleTypes {
    ...TypeRef
  }
}

fragment InputValue on __InputValue {
  name
  description
  type { ...TypeRef }
  defaultValue
}

"""
        + type_ref_fragment
    )


async def execute_graphql_query(
    endpoint_url: str,
    query: str,
    operation_name: str,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute a GraphQL query and return the response.

    Args:
        endpoint_url: URL of the GraphQL endpoint
        query: The GraphQL query to execute
        operation_name: The name of the operation
        headers: Optional HTTP headers for the request
        timeout: Timeout in seconds

    Returns:
        The parsed GraphQL response

    Raises:
        Exception: If the query execution fails

    """
    if headers is None:
        headers = {}

    # Ensure we have the proper content type for GraphQL
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    payload = {
        "query": query,
        "operationName": operation_name,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            endpoint_url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )

        # Raise an exception for HTTP errors
        response.raise_for_status()

        return response.json()


async def fetch_schema(
    endpoint_url: str,
    headers: dict[str, str] | None = None,
    output_path: str | None = "schema.json",
    verbose: bool = False,
    max_depth: int = DEFAULT_MAX_DEPTH,
    wrap_with_data: bool = False,
) -> dict[str, Any]:
    """Execute introspection query against the GraphQL API and save the schema.

    Args:
        endpoint_url: URL of the GraphQL endpoint
        headers: Optional HTTP headers for the request
        output_path: Path where to save the schema JSON file, or None to not save
        verbose: Whether to print verbose output
        max_depth: Maximum nesting depth for type references
        wrap_with_data: Whether to wrap the schema in a 'data' object for compatibility

    Returns:
        The GraphQL schema as a dictionary

    """
    if verbose:
        pass

    # Try with initial depth
    current_depth = max_depth
    success = False
    schema_data = None

    while not success and current_depth >= 1:
        try:
            if verbose:
                pass

            # Generate query with current depth using the official format
            query = generate_official_introspection_query(current_depth)

            result = await execute_graphql_query(
                endpoint_url=endpoint_url,
                query=query,
                operation_name="IntrospectionQuery",
                headers=headers,
            )

            if result.get("errors"):
                error_msg = "; ".join(
                    error.get("message", "Unknown error") for error in result["errors"]
                )

                # Check if the error is related to query depth
                if any(
                    keyword in error.get("message", "").lower()
                    for keyword in ["depth", "complex", "nested"]
                ):
                    if verbose:
                        pass

                    # Reduce depth and try again
                    current_depth -= 1
                    if current_depth < 1:
                        msg = f"Failed with all depth settings: {error_msg}"
                        raise Exception(msg)
                    continue

                msg = f"GraphQL errors: {error_msg}"
                raise Exception(msg)

            schema_data = result["data"]
            success = True

            if verbose and current_depth < max_depth:
                pass

        except Exception as e:
            if (
                any(keyword in str(e).lower() for keyword in ["depth", "complex", "nested"])
                and current_depth > 1
            ):
                current_depth -= 1
                if verbose:
                    pass
            else:
                raise

    if schema_data is None:
        msg = "Failed to fetch schema with any depth setting"
        raise Exception(msg)

    # Prepare the output data - wrap schema in 'data' object if requested
    # This is for compatibility with tools like generate_from_json.py
    output_data = {"data": schema_data} if wrap_with_data else schema_data

    # Save the schema to the output file if path is provided
    if output_path:
        # Format exactly like get-graphql-schema with 2 space indent
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        if verbose:
            pass

    return schema_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate GraphQL schema from a remote API using introspection",
    )

    parser.add_argument(
        "-e",
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"GraphQL API endpoint URL (default: {DEFAULT_ENDPOINT})",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file path for the schema (default: {DEFAULT_OUTPUT})",
    )

    parser.add_argument(
        "-a",
        "--auth",
        help="Authorization token (will be added as 'Bearer TOKEN' in headers)",
    )

    parser.add_argument(
        "--auth-type",
        default="Bearer",
        help="Authorization type to use with --auth (default: Bearer)",
    )

    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Maximum type nesting depth (default: {DEFAULT_MAX_DEPTH}, range: 1-9)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Format output as JSON (this is always true now to match official tool)",
    )

    parser.add_argument(
        "--compatibility",
        action="store_true",
        default=False,
        help="Wrap schema in 'data' object for compatibility with older tools (default: False)",
    )

    return parser.parse_args()


async def main() -> None:
    """Run the schema generation process."""
    args = parse_args()

    # Set up headers
    headers = {}
    if args.auth:
        headers["Authorization"] = f"{args.auth_type} {args.auth}"

    if args.verbose and args.auth:
        pass

    try:
        await fetch_schema(
            endpoint_url=args.endpoint,
            headers=headers,
            output_path=args.output,
            verbose=args.verbose,
            max_depth=args.depth,
            wrap_with_data=args.compatibility,
        )
    except Exception:
        sys.exit(1)


# Helper function for importing into other modules
def generate_schema(
    endpoint_url: str,
    output_path: str | None = "schema.json",
    auth_token: str | None = None,
    auth_type: str = "Bearer",
    max_depth: int = DEFAULT_MAX_DEPTH,
    verbose: bool = False,
    wrap_with_data: bool = False,
) -> dict[str, Any]:
    """Generate a GraphQL schema by executing an introspection query.
    This is a synchronous wrapper around the async functionality.

    Args:
        endpoint_url: The GraphQL API endpoint URL
        output_path: Path to save the schema JSON file, or None to not save
        auth_token: Optional authentication token
        auth_type: Authentication type (default: "Bearer")
        max_depth: Maximum type nesting depth (default: 7, range: 1-9)
        verbose: Whether to print verbose output
        wrap_with_data: Whether to wrap schema in 'data' object for compatibility with tools

    Returns:
        The GraphQL schema as a dictionary

    Raises:
        Exception: If the schema generation fails

    """
    headers = {}
    if auth_token:
        headers["Authorization"] = f"{auth_type} {auth_token}"

    try:
        return asyncio.run(
            fetch_schema(
                endpoint_url=endpoint_url,
                headers=headers,
                output_path=output_path,
                verbose=verbose,
                max_depth=max_depth,
                wrap_with_data=wrap_with_data,
            ),
        )
    except Exception:
        if verbose:
            pass
        raise


if __name__ == "__main__":
    asyncio.run(main())
