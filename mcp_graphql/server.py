import asyncio
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from logging import INFO, basicConfig, getLogger

import mcp
from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool
from pydantic import TypeAdapter

from mcp_graphql.api_types import Query

# Configurar logging
basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Manage server startup and shutdown lifecycle."""
    yield {}


server = Server("example-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    tools = []
    # Get all methods from Query class that have an 'args' parameter
    for query_name, method in inspect.getmembers(Query, predicate=inspect.isfunction):
        if query_name.startswith("_"):
            continue

        # Get the method signature
        sig = inspect.signature(method)
        if "args" in sig.parameters:
            # Get the type annotation for args
            args_type = sig.parameters["args"].annotation
            # Get JSON schema from Pydantic model
            schema = TypeAdapter(args_type).json_schema()

            # Use docstring as description if available, otherwise use default
            description = (
                method.__doc__.strip() if method.__doc__ else f"GraphQL query: {query_name}"
            )

            tools.append(
                Tool(
                    name=query_name,
                    description=description,
                    inputSchema=schema,
                ),
            )

    return tools


# Un único handler para todas las herramientas GraphQL
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    """Execute a GraphQL query based on the tool name"""
    # Importar dinámicamente las clases y funciones necesarias
    from mcp_graphql.generate_from_json import load_graphql_schema
    from mcp_graphql.graphql_client import GraphQLClient, GraphQLSchema

    # Intentar cargar el schema con generate_schema si está disponible
    schema_data = None
    try:
        # Primero intenta utilizar generate_schema.py si está disponible
        # ya que tiene manejo de errores y recuperación más robustos
        import mcp_graphql.generate_schema as generate_schema

        logger.info("Using generate_schema to fetch schema dynamically")
        try:
            schema_content = generate_schema.generate_schema(
                endpoint_url="http://localhost:3010/graphql",
                output_path=None,  # No guardar a archivo
                verbose=False,
                wrap_with_data=False,  # No envolver en data
            )
            # Envolver en el formato esperado por GraphQLSchema
            schema_data = {"__schema": schema_content}
            logger.info("Schema fetched dynamically")
        except Exception as e:
            logger.warning(f"Failed to fetch schema dynamically: {e!s}")
    except ImportError:
        logger.info("generate_schema module not available, checking for local schema file")
        # Cargar el schema local si existe
        import os

        schema_path = "schema.json" if os.path.exists("schema.json") else None
        if schema_path:
            try:
                # load_graphql_schema ya devuelve el contenido de __schema directamente
                schema_content = load_graphql_schema(schema_path)
                logger.info(f"Schema loaded from file: {schema_path}")
                # Necesitamos usar directamente este contenido como self.schema, no como schema_dict
                schema_data = {"__schema": schema_content}
            except Exception as e:
                logger.warning(f"Failed to load schema from file: {e!s}")

    # Crear el cliente GraphQL
    graphql_client = GraphQLClient(url="http://localhost:3010/graphql")
    if schema_data:
        graphql_client.schema = GraphQLSchema(schema_dict=schema_data)

    # Obtener los métodos dinámicamente
    # Importar la clase Query dinámicamente para no tener referencias directas
    from importlib import import_module
    from inspect import getmembers, isfunction

    query_module = import_module("api_types")
    Query = query_module.Query

    # Obtener todos los métodos de la clase Query
    methods = {
        method_name: method
        for method_name, method in getmembers(Query, predicate=isfunction)
        if not method_name.startswith("_")
    }

    # Verificar si el nombre corresponde a un método de Query
    if name not in methods:
        return [TextContent(type="text", text=f"Unknown query: {name}")]

    # Obtener el método a ejecutar
    method = methods[name]
    sig = inspect.signature(method)

    if "args" not in sig.parameters:
        return [TextContent(type="text", text=f"Method {name} does not accept args parameter")]

    # Obtener el tipo de los argumentos
    args_type = sig.parameters["args"].annotation

    try:
        # Crear instancia de los argumentos
        args_instance = args_type(**arguments)

        # Obtener las variables para la consulta
        variables = args_instance.model_dump(by_alias=True) if args_instance else None

        # Dejamos que el cliente genere los campos automáticamente
        # sin hacer referencia a nombres específicos
        result = graphql_client.execute_query(None, name, None, variables)

        # Mostrar el resultado en formato texto
        return [TextContent(type="text", text=f"Result for {name}: {result!s}")]
    except Exception as e:
        logger.error(f"Error executing {name} query: {e!s}", exc_info=True)
        return [TextContent(type="text", text=f"Error executing {name} query: {e!s}")]


async def run() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="example",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
