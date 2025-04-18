import contextlib
import json
import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def camel_case(snake_str: str) -> str:
    """Convert snake_case to CamelCase for GraphQL operations"""
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase for GraphQL operations"""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


class GraphQLSchema:
    """A class that represents a GraphQL schema and provides utilities for query generation"""

    def __init__(self, schema_dict: dict | None = None) -> None:
        if not schema_dict:
            self.schema = None
            self.types_by_name = {}
            return

        # Manejar diferentes formatos de schema
        if "__schema" in schema_dict:
            self.schema = schema_dict["__schema"]
        elif "data" in schema_dict and "__schema" in schema_dict["data"]:
            self.schema = schema_dict["data"]["__schema"]
        else:
            msg = "Invalid schema format"
            raise ValueError(msg, schema_dict)

        self.types_by_name = {}
        if self.schema:
            # Index types by name
            for t in self.schema.get("types", []):
                if t["name"] and not t["name"].startswith("__"):
                    self.types_by_name[t["name"]] = t

    def get_field_type(self, type_obj, non_null=False):
        """Extract the base type name from a type object"""
        if type_obj.get("kind") == "NON_NULL":
            return self.get_field_type(type_obj["ofType"], True)
        if type_obj.get("kind") == "LIST":
            inner_type = self.get_field_type(type_obj["ofType"])
            return f"[{inner_type}]"
        return type_obj.get("name")

    def get_query_fields(self, type_name: str, depth: int = 2, visited_types=None) -> str:
        """Generate a GraphQL query fields string based on the schema"""
        if depth <= 0 or not type_name or type_name not in self.types_by_name:
            return "id"

        # Usar un conjunto para llevar el registro de los tipos ya visitados y evitar ciclos
        if visited_types is None:
            visited_types = set()

        # Si este tipo ya ha sido visitado, manejar de forma inteligente
        if type_name in visited_types:
            # En lugar de devolver un conjunto fijo de campos,
            # analizar el tipo y devolver los campos escalares disponibles
            type_obj = self.types_by_name[type_name]
            available_fields = []

            # Buscar campos escalares seguros para incluir
            for field in type_obj.get("fields", []):
                field_name = field["name"]
                field_type = field.get("type", {})

                # Extraer el tipo base
                kind = None
                while field_type:
                    kind = field_type.get("kind")
                    if kind in {"NON_NULL", "LIST"}:
                        field_type = field_type.get("ofType", {})
                    else:
                        break

                # Incluir solo campos escalares para evitar ciclos infinitos
                if kind == "SCALAR":
                    # Siempre incluir id si existe
                    if field_name == "id":
                        return "id"  # Priorizar id si existe
                    available_fields.append(field_name)

            # Devolver todos los campos escalares disponibles o un subconjunto si hay muchos
            if available_fields:
                # Si hay muchos campos, priorizar algunos comunes
                priority_fields = ["cursor", "endCursor", "hasNextPage", "hasPreviousPage"]
                for field in priority_fields:
                    if field in available_fields:
                        return field  # Devolver un campo prioritario si existe

                # Limitar a un máximo de 3-4 campos escalares para evitar consultas muy grandes
                if len(available_fields) > 4:
                    available_fields = available_fields[:4]

                return " ".join(available_fields)

            # Si no se encuentran campos escalares, devolver vacío
            return ""

        # Marcar este tipo como visitado
        visited_types.add(type_name)

        type_obj = self.types_by_name[type_name]

        # Handle scalar types
        if type_obj["kind"] == "SCALAR":
            return ""

        # Obtener todos los nombres de campo disponibles para este tipo
        available_fields = {f["name"] for f in type_obj.get("fields", [])}

        # Handle object and interface types
        fields = []
        for field in type_obj.get("fields", []):
            field_name = field["name"]
            field_type_obj = field["type"]
            field_type_name = self.get_field_type(field_type_obj)

            # Skip fields that require arguments
            if field.get("args") and any(
                arg["type"].get("kind") == "NON_NULL" for arg in field["args"]
            ):
                continue

            # Por seguridad, limitar ciertos campos que podrían causar consultas muy grandes
            if depth == 1 and field_name.lower() in [
                "edges",
                "nodes",
                "items",
                "connections",
                "friends",
                "followers",
            ] and field_type_name.startswith("["):
                continue

            # For scalar fields, just add the field name
            if field_type_name in ["String", "Int", "Float", "Boolean", "ID"]:
                fields.append(field_name)
            # For object fields, recursively get the fields
            elif field_type_name in self.types_by_name:
                # Check if it's an object type
                inner_type = self.types_by_name[field_type_name]
                if inner_type["kind"] in ["OBJECT", "INTERFACE"]:
                    # Si es una consulta de conexión o lista que podría ser grande, limitar profundidad
                    nested_depth = depth - 1
                    if field_name.lower().endswith(("connection", "edge", "edges", "nodes")):
                        nested_depth = min(nested_depth, 1)

                    # Recursively get fields for nested objects
                    nested_fields = self.get_query_fields(
                        field_type_name, nested_depth, visited_types.copy(),
                    )
                    if nested_fields:
                        fields.append(f"{field_name} {{ {nested_fields} }}")
                    else:
                        fields.append(field_name)
            # Handle list types
            elif field_type_name.startswith("[") and field_type_name.endswith("]"):
                inner_type_name = field_type_name[1:-1]
                if inner_type_name in self.types_by_name:
                    # Limitar aún más la profundidad para listas anidadas
                    nested_depth = depth - 1
                    if field_name.lower().endswith(("s", "connection", "list", "collection")):
                        nested_depth = min(nested_depth, 1)

                    nested_fields = self.get_query_fields(
                        inner_type_name, nested_depth, visited_types.copy(),
                    )
                    if nested_fields:
                        fields.append(f"{field_name} {{ {nested_fields} }}")
                    else:
                        fields.append(field_name)
                else:
                    fields.append(field_name)

        # Remover visited_types para evitar problemas de memoria
        visited_types.remove(type_name)

        return " ".join(fields)


class GraphQLClient:
    """Base class for GraphQL clients that handles HTTP requests"""

    def __init__(self, url: str | None = None, schema_dict: dict | None = None) -> None:
        self.url = url or "http://localhost:4000/graphql"
        self.headers = {"Content-Type": "application/json"}
        self.schema = None
        if schema_dict:
            with contextlib.suppress(Exception):
                self.schema = GraphQLSchema(schema_dict=schema_dict)

    def execute(self, query: str, variables: dict | None = None, operation_name: str | None = None) -> dict:
        """Execute a GraphQL query or mutation"""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        if operation_name:
            payload["operationName"] = operation_name

        with httpx.Client() as client:
            response = client.post(self.url, json=payload, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                msg = f"GraphQL Error: {json.dumps(result['errors'])}"
                raise Exception(msg)
            return result

    def execute_query(
        self,
        model_class: type[T] | str | None,
        query_name: str,
        query_fields: str | None = None,
        variables: dict | None = None,
        depth: int = 1,  # Reducir profundidad predeterminada para evitar consultas muy grandes
    ) -> Any:
        """Execute a GraphQL query and return the typed result"""
        # If query_fields is not provided, try to generate them from schema
        if query_fields is None:
            if self.schema:
                # Try to find the type in schema based on the query name
                schema_type_name = None
                query_type = self.schema.types_by_name.get("Query")
                if query_type:
                    for field in query_type.get("fields", []):
                        if field["name"] == query_name:
                            response_type = field["type"]
                            # Handle non-null wrapper
                            if response_type["kind"] == "NON_NULL":
                                response_type = response_type["ofType"]
                            schema_type_name = response_type.get("name")
                            break

                if schema_type_name:
                    # Usar la profundidad especificada o la predeterminada
                    query_fields = self.schema.get_query_fields(schema_type_name, depth)
                    # Si model_class es None, asignarle el nombre del tipo del schema
                    if model_class is None:
                        model_class = schema_type_name
                else:
                    # Fallback to id field
                    query_fields = "id"
            else:
                # Fallback to id field
                query_fields = "id"

        # Generar la query sin envolver variables en un objeto args
        var_declarations = []
        var_in_query = []

        if variables:
            # Try to get arg types from schema
            arg_types = {}
            if self.schema:
                query_type = self.schema.types_by_name.get("Query")
                if query_type:
                    for field in query_type.get("fields", []):
                        if field["name"] == query_name:
                            for arg in field.get("args", []):
                                arg_name = arg["name"]
                                arg_type = self.schema.get_field_type(arg["type"])
                                arg_required = arg["type"]["kind"] == "NON_NULL"
                                arg_types[arg_name] = (arg_type, arg_required)

            for key, value in variables.items():
                # Convertir snake_case a camelCase para las variables
                camel_key = key if "_" not in key else snake_to_camel(key)

                # Try to get type from schema
                if camel_key in arg_types:
                    var_type, required = arg_types[camel_key]
                else:
                    # Determine type based on value (simplified)
                    var_type = "ID" if key == "id" else "String"
                    if isinstance(value, int):
                        var_type = "Int"
                    elif isinstance(value, float):
                        var_type = "Float"
                    elif isinstance(value, bool):
                        var_type = "Boolean"

                    # Add exclamation mark for non-null values
                    required = value is not None

                if required:
                    var_type += "!"

                var_declarations.append(f"${camel_key}: {var_type}")
                var_in_query.append(f"{camel_key}: ${camel_key}")

        # Construir la query
        if var_declarations:
            query = f"""query {query_name}({", ".join(var_declarations)}) {{
            {query_name}({", ".join(var_in_query)}) {{
                {query_fields}
            }}
        }}"""
        else:
            query = f"""query {query_name} {{
            {query_name} {{
                {query_fields}
            }}
        }}"""

        # Para debugging
        LOGGER.debug(f"Generated query: {query}")
        LOGGER.debug(f"Variables: {variables}")

        try:
            result = self.execute(query, variables)
            data = result["data"][query_name]
            LOGGER.debug(f"Query result: {result}")

            # Simplemente devolver los datos si no hay clase modelo
            if model_class is None:
                return data

            # Handle different ways of model instantiation
            if isinstance(model_class, str):
                # Just return the data if model_class is a string
                return data  # type: ignore
            if hasattr(model_class, "model_validate"):
                return model_class.model_validate(data)
            if hasattr(model_class, "parse_obj"):
                return model_class.parse_obj(data)
            return model_class(**data)  # type: ignore
        except Exception as e:
            # Log the error with the query that caused it
            LOGGER.exception(f"Error executing query: {e}")
            LOGGER.exception(f"Query: {query}")
            LOGGER.exception(f"Variables: {variables}")
            # Re-raise the exception so it can be handled by the caller
            raise

    def execute_mutation(
        self,
        model_class: type[T] | str | None,
        mutation_name: str,
        mutation_fields: str | None = None,
        variables: dict | None = None,
        depth: int = 1,  # Reducir profundidad predeterminada para evitar consultas muy grandes
    ) -> Any:
        """Execute a GraphQL mutation and return the typed result"""
        # If mutation_fields is not provided, try to generate them from schema
        if mutation_fields is None:
            if self.schema:
                # Try to find the type in schema based on the mutation name
                schema_type_name = None
                mutation_type = self.schema.types_by_name.get("Mutation")
                if mutation_type:
                    for field in mutation_type.get("fields", []):
                        if field["name"] == mutation_name:
                            response_type = field["type"]
                            # Handle non-null wrapper
                            if response_type["kind"] == "NON_NULL":
                                response_type = response_type["ofType"]
                            schema_type_name = response_type.get("name")
                            break

                if schema_type_name:
                    # Usar la profundidad especificada o la predeterminada
                    mutation_fields = self.schema.get_query_fields(schema_type_name, depth)
                    # Si model_class es None, asignarle el nombre del tipo del schema
                    if model_class is None:
                        model_class = schema_type_name
                else:
                    # Fallback to id field
                    mutation_fields = "id"
            else:
                # Fallback to id field
                mutation_fields = "id"

        # Generar la mutation sin envolver variables en un objeto args
        var_declarations = []
        var_in_mutation = []

        if variables:
            # Try to get arg types from schema
            arg_types = {}
            if self.schema:
                mutation_type = self.schema.types_by_name.get("Mutation")
                if mutation_type:
                    for field in mutation_type.get("fields", []):
                        if field["name"] == mutation_name:
                            for arg in field.get("args", []):
                                arg_name = arg["name"]
                                arg_type = self.schema.get_field_type(arg["type"])
                                arg_required = arg["type"]["kind"] == "NON_NULL"
                                arg_types[arg_name] = (arg_type, arg_required)

            for key, value in variables.items():
                # Convertir snake_case a camelCase para las variables
                camel_key = key if "_" not in key else snake_to_camel(key)

                # Try to get type from schema
                if camel_key in arg_types:
                    var_type, required = arg_types[camel_key]
                else:
                    # Determine type based on value (simplified)
                    var_type = "ID" if key == "id" else "String"
                    if isinstance(value, int):
                        var_type = "Int"
                    elif isinstance(value, float):
                        var_type = "Float"
                    elif isinstance(value, bool):
                        var_type = "Boolean"

                    # Add exclamation mark for non-null values
                    required = value is not None

                if required:
                    var_type += "!"

                var_declarations.append(f"${camel_key}: {var_type}")
                var_in_mutation.append(f"{camel_key}: ${camel_key}")

        # Construir la mutation
        if var_declarations:
            mutation = f"""mutation {mutation_name}({", ".join(var_declarations)}) {{
            {mutation_name}({", ".join(var_in_mutation)}) {{
                {mutation_fields}
            }}
        }}"""
        else:
            mutation = f"""mutation {mutation_name} {{
            {mutation_name} {{
                {mutation_fields}
            }}
        }}"""

        # Para debugging
        LOGGER.debug(f"Generated mutation: {mutation}")
        LOGGER.debug(f"Variables: {variables}")

        try:
            result = self.execute(mutation, variables)
            data = result["data"][mutation_name]
            LOGGER.debug(f"Mutation result: {result}")

            # Simplemente devolver los datos si no hay clase modelo
            if model_class is None:
                return data

            # Handle different ways of model instantiation
            if isinstance(model_class, str):
                # Just return the data if model_class is a string
                return data  # type: ignore
            if hasattr(model_class, "model_validate"):
                return model_class.model_validate(data)
            if hasattr(model_class, "parse_obj"):
                return model_class.parse_obj(data)
            return model_class(**data)  # type: ignore
        except Exception as e:
            # Log the error with the mutation that caused it
            LOGGER.exception(f"Error executing mutation: {e}")
            LOGGER.exception(f"Mutation: {mutation}")
            LOGGER.exception(f"Variables: {variables}")
            # Re-raise the exception so it can be handled by the caller
            raise

    def get_model_fields(self, model_class: type[BaseModel], depth: int = 2) -> str:
        """Automatically generate GraphQL field selection based on Pydantic model"""
        if depth <= 0:
            return ""

        # Handle different ways to access model fields
        model_field_dict = None
        if hasattr(model_class, "model_fields"):  # Pydantic v2
            model_field_dict = model_class.model_fields
        elif hasattr(model_class, "__fields__"):  # Pydantic v1
            model_field_dict = model_class.__fields__

        if not model_field_dict:
            return "id name"

        try:
            # Get the model fields
            fields = []

            for field_name, field in model_field_dict.items():
                # Skip private fields
                if field_name.startswith("_"):
                    continue

                # Use the alias if it exists
                alias_attr = "alias" if hasattr(field, "alias") else "alias_priority"
                graphql_name = getattr(field, alias_attr, None) or field_name

                # Handle nested types - this is different between Pydantic v1 and v2
                field_type = getattr(field, "annotation", getattr(field, "type_", None))

                if self._is_model_class(field_type) and depth > 1:
                    nested_fields = self.get_model_fields(field_type, depth - 1)
                    fields.append(f"{graphql_name} {{ {nested_fields} }}")
                elif self._is_list_type(field_type):
                    # Handle List types
                    item_type = self._get_list_item_type(field_type)
                    if self._is_model_class(item_type) and depth > 1:
                        nested_fields = self.get_model_fields(item_type, depth - 1)
                        fields.append(f"{graphql_name} {{ {nested_fields} }}")
                    else:
                        fields.append(graphql_name)
                else:
                    fields.append(graphql_name)

            return " ".join(fields)
        except Exception:
            # If anything goes wrong, return a reasonable default
            return "id name"

    def _is_model_class(self, type_obj: Any) -> bool:
        """Check if a type is a Pydantic model class"""
        return (
            hasattr(type_obj, "model_fields")
            or hasattr(type_obj, "__fields__")
            or (hasattr(type_obj, "__origin__") and type_obj.__origin__ is type)
        )

    def _is_list_type(self, type_obj: Any) -> bool:
        """Check if a type is a List type"""
        return hasattr(type_obj, "__origin__") and type_obj.__origin__ in (list, list)

    def _get_list_item_type(self, list_type: Any) -> Any:
        """Extract the item type from a List type"""
        if hasattr(list_type, "__args__") and list_type.__args__:
            return list_type.__args__[0]
        return Any
