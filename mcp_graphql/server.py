import functools
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from logging import INFO, basicConfig, getLogger
from typing import TYPE_CHECKING, Any, cast

from gql import Client
from gql.dsl import DSLField, DSLQuery, DSLSchema, DSLType, GraphQLObjectType, dsl_gql
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import (
    GraphQLInputType,
    GraphQLInterfaceType,
    GraphQLList,
    GraphQLNonNull,
    GraphQLOutputType,
    GraphQLScalarType,
    print_ast,
)
from mcp import types as mcp_types
from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from mcp_graphql.types import (
    JsonSchema,
    NestedSelection,
    ProcessedNestedType,
    QueryTypeNotFoundError,
    SchemaRetrievalError,
    ServerContext,
)

if TYPE_CHECKING:
    from graphql import GraphQLArgumentMap, GraphQLField

# Configurar logging
basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server[ServerContext], api_url: str, auth_headers: dict[str, str]) -> AsyncIterator[ServerContext]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize resources on startup
    transport = AIOHTTPTransport(url=api_url, headers=auth_headers)
    client = Client(transport=transport, fetch_schema_from_transport=True)
    # Use the client directly instead of trying to use session as a context manager
    async with client as session:
        try:
            session.client
            context: ServerContext = {"session": session, "dsl_schema": DSLSchema(session.client.schema)}
            yield context
        finally:
            # No need for manual __aexit__ call - it's handled by the async with
            pass


def convert_type_to_json_schema(
    gql_type: GraphQLInputType,
    max_depth: int = 3,
    current_depth: int = 1,
) -> JsonSchema:
    """
    Convert GraphQL type to JSON Schema, handling complex nested types properly.
    Supports max_depth to prevent infinite recursion with circular references.
    """
    # Check max depth to prevent infinite recursion
    if current_depth > max_depth:
        return {"type": "object", "description": "Max depth reached"}

    # Handle Non-Null types
    if isinstance(gql_type, GraphQLNonNull):
        inner_schema = convert_type_to_json_schema(gql_type.of_type, max_depth, current_depth)
        # Mark this as required via the flag (will be processed by the caller)
        inner_schema["required"] = True  # type: ignore
        return inner_schema

    # Handle List types
    if isinstance(gql_type, GraphQLList):
        inner_schema = convert_type_to_json_schema(gql_type.of_type, max_depth, current_depth)
        return {"type": "array", "items": inner_schema}

    # Handle scalar types based on name
    if isinstance(gql_type, GraphQLScalarType):
        type_name = str(gql_type).lower()
        if type_name == "string":
            return {"type": "string"}
        if type_name == "int":
            return {"type": "integer"}
        if type_name == "float":
            return {"type": "number"}
        if type_name == "boolean":
            return {"type": "boolean"}
        if type_name in ["id", "id!"]:
            return {"type": "string"}
        # Generic scalar (DateTime, etc)
        return {"type": "string", "description": f"GraphQL scalar: {gql_type!s}"}

    # Handle Object types and Input Object types
    if hasattr(gql_type, "fields"):
        # Create an object type with properties
        properties = {}
        required = []

        # Process each field
        for field_name, field_value in gql_type.fields.items():
            # Skip internal fields
            if field_name.startswith("__"):
                continue

            # Get field type schema
            field_schema = convert_type_to_json_schema(
                field_value.type,
                max_depth,
                current_depth + 1,
            )

            # Check if field is required
            is_required = field_schema.pop("required", False)
            if is_required:
                required.append(field_name)

            # Add field schema to properties
            properties[field_name] = field_schema

        # Construct object schema
        object_schema = {
            "type": "object",
            "properties": properties,
        }

        # Add required array if needed
        if required:
            object_schema["required"] = cast("Any", required)  # Force cast to Any to bypass type checking

        return cast("JsonSchema", object_schema)

    # Fallback for other types
    type_name = str(gql_type)
    logger.info("Unknown GraphQL type: %s, using string fallback", type_name)
    return {"type": "string", "description": f"Unknown GraphQL type: {type_name}"}


def _process_nested_type(
    field_name: str,
    nested_type: GraphQLOutputType,
    max_depth: int,
    current_depth: int,
) -> ProcessedNestedType:
    """Process a nested type field."""
    # Handle non-null and list wrappers
    while hasattr(nested_type, "of_type"):
        nested_type = nested_type.of_type

    # Only process if we actually have a GraphQLObjectType
    if isinstance(nested_type, GraphQLObjectType):
        nested_selections = build_nested_selection(
            nested_type,
            max_depth,
            current_depth + 1,
        )
        # Only append if there are valid nested selections
        if nested_selections:
            return (field_name, nested_selections)
    return (field_name, None)  # Return properly typed tuple instead of None


def build_nested_selection(
    field_type: GraphQLObjectType | GraphQLInterfaceType,
    max_depth: int,
    current_depth: int = 1,
) -> NestedSelection:
    """Recursively build nested selections up to the specified depth."""
    # Early return if max depth reached
    if current_depth > max_depth:
        return []

    # Check if type is an Enum or other type without fields
    if not hasattr(field_type, "fields"):
        # For enum types or other types without fields, we can't select sub-fields
        return []

    selections: NestedSelection = []
    for field_name, field_value in cast("dict[str, GraphQLField]", field_type.fields).items():
        # Skip internal fields (starting with __)
        if field_name.startswith("__"):
            continue

        # Handle different field types
        if isinstance(field_value.type, GraphQLScalarType): # type: ignore
            selections.append((field_name, None))
        elif isinstance(field_value.type, GraphQLNonNull):
            of_type = field_value.type.of_type
            # Check if field is a scalar
            if isinstance(of_type, GraphQLScalarType):
                # Add scalar field to selections
                selections.append((field_name, None))
            else:
                result = _process_nested_type(
                    field_name, field_value.type, max_depth, current_depth,
                )
                if result:
                    selections.append(result)
        elif isinstance(field_value.type, GraphQLList) or not isinstance(
            field_value.type, GraphQLScalarType,
        ):
            result = _process_nested_type(field_name, field_value.type, max_depth, current_depth)
            if result:
                selections.append(result)

    return selections


def build_selection(ds: DSLSchema, parent: DSLType, selections: Any) -> list[DSLField]:
    result = []
    for field_name, nested_selections in selections:
        # Get the field
        field = getattr(parent, field_name)

        # Get the field type and handle wrapped types (List, NonNull)
        field_type = field.field.type
        # Unwrap NonNull and List types to get the inner type
        while hasattr(field_type, "of_type"):
            field_type = field_type.of_type

        # Check if this is a scalar type or an object type
        is_scalar = isinstance(field_type, GraphQLScalarType)

        if nested_selections is None and is_scalar:
            # This is a scalar field - can be selected directly
            result.append(getattr(parent, field_name))
        elif nested_selections and len(nested_selections) > 0:
            # This is a non-scalar with valid nested selections
            nested_fields = build_selection(ds, getattr(ds, field_type.name), nested_selections)
            if nested_fields:
                result.append(field.select(*nested_fields))
        # Skip fields that have no valid nested selections and aren't scalars

    return result


async def list_tools_impl(_server: Server[ServerContext]) -> list[Tool]:
    try:
        ctx = _server.request_context
        ds: DSLSchema = ctx.lifespan_context["dsl_schema"]
    except LookupError as e:
        logger.info("Error al obtener el contexto: %s", e)
        # Configura el transporte
        transport = AIOHTTPTransport(url="http://localhost:8080/graphql")

        # Crea el cliente con fetch_schema_from_transport=True
        client = Client(transport=transport, fetch_schema_from_transport=True)
        async with client as session:
            if not session.client.schema:
                raise SchemaRetrievalError
            ds = DSLSchema(session.client.schema)
    tools: list[Tool] = []

    # Establece la sesión del cliente
    if ds:
        # Accede al esquema dentro de la sesión
        if not ds._schema.query_type:
            raise QueryTypeNotFoundError
        fields: dict[str, GraphQLField] = ds._schema.query_type.fields
        for query_name, field in fields.items():
            args_map: GraphQLArgumentMap = field.args
            args_schema: JsonSchema = {"type": "object", "properties": {}, "required": []}
            for arg_name, arg in args_map.items():
                logger.info("Converting GraphQL type for %s: %s", arg_name, arg.type.name) # type: ignore
                type_schema = convert_type_to_json_schema(arg.type, max_depth=3, current_depth=1)
                # Remove the "required" flag which was used for tracking
                is_required = type_schema.pop("required", False)

                args_schema["properties"][arg_name] = type_schema
                args_schema["properties"][arg_name]["description"] = (
                    arg.description if arg.description else f"Argument {arg_name}"
                )

                # Mark as required if non-null and no default value
                if (is_required or str(arg.type).startswith("!")) and not arg.default_value:
                    if not isinstance(args_schema["required"], bool):
                        args_schema["required"].append(arg_name)
            logger.info("args_schema: %s", json.dumps(args_schema, indent=2))

            tools.append(
                Tool(
                    name=query_name,
                    description=field.description
                    if field.description
                    else f"GraphQL query: {query_name}",
                    inputSchema=args_schema, # type: ignore
                ),
            )

    return tools


async def call_tool_impl(_server: Server[ServerContext], name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
    ctx = _server.request_context
    session = ctx.lifespan_context["session"]
    # Don't use the session as a context manager, use it directly
    ds: DSLSchema = ctx.lifespan_context["dsl_schema"]
    if not ds._schema.query_type:
        raise QueryTypeNotFoundError
    fields: dict[str, GraphQLField] = ds._schema.query_type.fields

    # Get query depth from arguments, default to 1 (flat)
    max_depth = 3
    try:
        max_depth = int(max_depth)
    except (ValueError, TypeError):
        max_depth = 1
    logger.info("Llamando a la herramienta %s con argumentos %s", name, arguments)
    if _query_name := next((_query_name for _query_name in fields if _query_name == name), None):
        attr: DSLField = getattr(ds.Query, _query_name)

        # Unwrap the type (NonNull, List) to get to the actual type name
        field_type = attr.field.type
        # Keep unwrapping until we find a type with a name attribute
        while hasattr(field_type, "of_type") and not hasattr(field_type, "name"):
            field_type = field_type.of_type

        # Now we should have the actual type with a name
        if not hasattr(field_type, "name"):
            return [
                mcp_types.TextContent(
                    type="text",
                    text=f"Error: No se pudo determinar el tipo de retorno para {name}",
                ),
            ]

        return_type: DSLType = getattr(ds, field_type.name)

        # Build the query with nested selections
        selections = build_nested_selection(return_type._type, max_depth)

        # Build the actual query
        query_selections = build_selection(ds, return_type, selections)
        query = dsl_gql(DSLQuery(attr(**arguments).select(*query_selections)))
        logger.info("query: %s", print_ast(query))

        #     # Execute the query
        result = await session.execute(query)
        return [mcp_types.TextContent(type="text", text=json.dumps(result))]

    # Error case - tool not found
    return [mcp_types.TextContent(type="text", text="No se encontró la herramienta")]


async def serve(api_url: str, auth_headers: dict[str, str] | None) -> None:
    server = Server[ServerContext](
        "mcp-graphql",
        lifespan=partial(server_lifespan, api_url=api_url, auth_headers=auth_headers or {}),
    )

    server.list_tools()(functools.partial(list_tools_impl, server))
    server.call_tool()(functools.partial(call_tool_impl, server)) # type: ignore

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-graphql",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
