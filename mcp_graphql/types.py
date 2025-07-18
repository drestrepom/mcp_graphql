from typing import Any, TypedDict

from gql.client import AsyncClientSession
from gql.dsl import DSLSchema


class ServerContext(TypedDict):
    """TypedDict for the server lifespan context."""

    session: AsyncClientSession
    dsl_schema: DSLSchema


class SchemaRetrievalError(Exception):
    """Exception raised when the GraphQL schema cannot be retrieved."""

    def __init__(self, message: str = "Failed to retrieve GraphQL schema") -> None:
        self.message = message
        super().__init__(self.message)


class QueryTypeNotFoundError(Exception):
    """Exception raised when the GraphQL schema doesn't have a query type."""

    def __init__(self, message: str = "GraphQL schema does not contain a query type") -> None:
        self.message = message
        super().__init__(self.message)


# Simplified JSON Schema TypedDict that matches the actual implementation
class JsonSchema(TypedDict, total=False):
    """TypedDict for JSON Schema used in GraphQL type conversion."""

    type: str
    description: str
    properties: dict[str, "JsonSchema"]
    required: bool | list[str]
    items: "JsonSchema"
    oneOf: list[dict[str, str]] | None


# Create simpler flattened definition for nested selections to avoid complex recursive types
NestedSelectionItem = tuple[str, list[Any] | None]
NestedSelection = list[NestedSelectionItem]

# ProcessedNestedType is the same as a NestedSelectionItem
ProcessedNestedType = NestedSelectionItem
