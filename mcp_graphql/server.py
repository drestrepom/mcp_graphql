import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
from logging import INFO, basicConfig, getLogger
from typing import TYPE_CHECKING

from graphql import GraphQLInputType, GraphQLList, GraphQLNonNull, GraphQLScalarType, print_ast
import mcp
from gql import Client
from gql.dsl import DSLField, DSLQuery, DSLSchema, dsl_gql, DSLType, GraphQLObjectType
from gql.transport.aiohttp import AIOHTTPTransport
from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import Tool
from mcp.types import TextContent, Tool
import mcp.types as types
import functools
from functools import partial

if TYPE_CHECKING:
    from graphql import GraphQLArgumentMap, GraphQLField

# Configurar logging
basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server, api_url: str, auth_headers: dict) -> AsyncIterator[dict]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize resources on startup
    transport = AIOHTTPTransport(url=api_url, headers=auth_headers)
    client = Client(transport=transport, fetch_schema_from_transport=True)
    # Use the client directly instead of trying to use session as a context manager
    async with client as session:
        try:
            yield {"session": session, "dsl_schema": DSLSchema(session.client.schema)}
        finally:
            # No need for manual __aexit__ call - it's handled by the async with
            pass


def convert_type_to_json_schema(
    gql_type: GraphQLInputType, max_depth: int = 3, current_depth: int = 1
):
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
        inner_schema["required"] = True
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
        elif type_name == "int":
            return {"type": "integer"}
        elif type_name == "float":
            return {"type": "number"}
        elif type_name == "boolean":
            return {"type": "boolean"}
        elif type_name in ["id", "id!"]:
            return {"type": "string"}
        else:
            # Generic scalar (DateTime, etc)
            return {"type": "string", "description": f"GraphQL scalar: {str(gql_type)}"}

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
                field_value.type, max_depth, current_depth + 1
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
            object_schema["required"] = required

        return object_schema

    # Fallback for other types
    type_name = str(gql_type)
    logger.info(f"Unknown GraphQL type: {type_name}, using string fallback")
    return {"type": "string", "description": f"Unknown GraphQL type: {type_name}"}


def build_nested_selection(field_type: GraphQLObjectType, max_depth: int, current_depth: int = 1):
    """Recursively build nested selections up to the specified depth."""
    # Early return if max depth reached
    if current_depth > max_depth:
        return []

    # Check if type is an Enum or other type without fields
    if not hasattr(field_type, "fields"):
        # For enum types or other types without fields, we can't select sub-fields
        return []

    selections = []
    for field_name, field_value in field_type.fields.items():
        # Skip internal fields (starting with __)
        if field_name.startswith("__"):
            continue
        if isinstance(field_value.type, GraphQLScalarType):
            selections.append((field_name, None))
        elif isinstance(field_value.type, GraphQLNonNull):
            of_type = field_value.type.of_type
            # Check if field is a scalar
            is_scalar = isinstance(of_type, GraphQLScalarType)
            if is_scalar:
                # Add scalar field to selections
                selections.append((field_name, None))
            else:
                # Get the nested type
                nested_type = field_value.type
                # Handle non-null and list wrappers
                while hasattr(nested_type, "of_type"):
                    nested_type = nested_type.of_type

                # Only add non-scalar fields with sub-selections
                nested_selections = build_nested_selection(
                    nested_type, max_depth, current_depth + 1
                )
                # Only add if it has valid nested selections
                if nested_selections:
                    selections.append((field_name, nested_selections))
        elif isinstance(field_value.type, GraphQLList):
            # Get the nested type
            nested_type = field_value.type
            # Handle non-null and list wrappers
            while hasattr(nested_type, "of_type"):
                nested_type = nested_type.of_type

            # Only process if we actually have a GraphQLObjectType
            if isinstance(nested_type, GraphQLObjectType):
                nested_selections = build_nested_selection(
                    nested_type, max_depth, current_depth + 1
                )
                # Only append if there are valid nested selections
                if nested_selections:
                    selections.append((field_name, nested_selections))
        else:
            # Get the nested type
            nested_type = field_value.type
            # Handle non-null and list wrappers
            while hasattr(nested_type, "of_type"):
                nested_type = nested_type.of_type

            # Only process if we actually have a GraphQLObjectType
            if isinstance(nested_type, GraphQLObjectType):
                nested_selections = build_nested_selection(
                    nested_type, max_depth, current_depth + 1
                )
                # Only append if there are valid nested selections
                if nested_selections:
                    selections.append((field_name, nested_selections))

    return selections


def build_selection(ds: DSLSchema, parent, selections):
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


async def list_tools_impl(_server: Server) -> list[Tool]:
    try:
        ctx = _server.request_context
        ds: DSLSchema = ctx.lifespan_context["dsl_schema"]
    except LookupError as e:
        logger.info(f"Error al obtener el contexto: {e}")
        # Configura el transporte
        transport = AIOHTTPTransport(url="http://localhost:8080/graphql")

        # Crea el cliente con fetch_schema_from_transport=True
        client = Client(transport=transport, fetch_schema_from_transport=True)
        async with client as session:
            ds = DSLSchema(session.client.schema)
    tools = []

    # Establece la sesión del cliente
    if ds:
        # Accede al esquema dentro de la sesión
        fields: dict[str, GraphQLField] = ds._schema.query_type.fields
        for query_name, field in fields.items():
            args_map: GraphQLArgumentMap = field.args
            args_schema = {"type": "object", "properties": {}, "required": []}
            for arg_name, arg in args_map.items():
                logger.info(f"Converting GraphQL type for {arg_name}: {str(arg.type)}")
                type_schema = convert_type_to_json_schema(arg.type, max_depth=3, current_depth=1)
                # Remove the "required" flag which was used for tracking
                is_required = type_schema.pop("required", False)

                args_schema["properties"][arg_name] = type_schema
                args_schema["properties"][arg_name]["description"] = (
                    arg.description if arg.description else f"Argument {arg_name}"
                )

                # Mark as required if non-null and no default value
                if (is_required or str(arg.type).startswith("!")) and not arg.default_value:
                    args_schema["required"].append(arg_name)
            logger.info(f"args_schema: {json.dumps(args_schema, indent=2)}")

            tools.append(
                Tool(
                    name=query_name,
                    description=field.description
                    if field.description
                    else f"GraphQL query: {query_name}",
                    inputSchema=args_schema,
                ),
            )

    return tools


async def call_tool_impl(_server: Server, name: str, arguments: dict) -> list:
    ctx = _server.request_context
    session = ctx.lifespan_context["session"]
    # Don't use the session as a context manager, use it directly
    ds: DSLSchema = ctx.lifespan_context["dsl_schema"]
    fields: dict[str, GraphQLField] = ds._schema.query_type.fields

    # Get query depth from arguments, default to 1 (flat)
    max_depth = arguments.pop("depth", 3) if arguments else 1
    try:
        max_depth = int(max_depth)
    except (ValueError, TypeError):
        max_depth = 1
    logger.info(f"Llamando a la herramienta {name} con argumentos {arguments}")
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
                types.TextContent(
                    type="text",
                    text=f"Error: No se pudo determinar el tipo de retorno para {name}",
                )
            ]

        return_type: DSLType = getattr(ds, field_type.name)

        # Build the query with nested selections
        selections = build_nested_selection(return_type._type, max_depth)

        # Build the actual query
        query_selections = build_selection(ds, return_type, selections)
        query = dsl_gql(DSLQuery(attr(**arguments).select(*query_selections)))
        logger.info(f"query: {print_ast(query)}")

        #     # Execute the query
        result = await session.execute(query)
        return [types.TextContent(type="text", text=json.dumps(result))]

    # Error case - tool not found
    return [types.TextContent(type="text", text="No se encontró la herramienta")]


async def serve(api_url: str, auth_headers: dict | None) -> None:
    server = Server(
        "mcp-graphql",
        lifespan=partial(server_lifespan, api_url=api_url, auth_headers=auth_headers or {}),
    )

    server.list_tools()(functools.partial(list_tools_impl, server))
    server.call_tool()(functools.partial(call_tool_impl, server))

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
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
