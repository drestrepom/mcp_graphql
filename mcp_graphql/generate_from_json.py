import json
import re
import sys
from typing import Any


def load_graphql_schema(schema_path: str) -> dict[str, Any]:
    """Load the GraphQL schema from JSON file."""
    # Try different encodings since some tools produce BOM-encoded files
    encodings_to_try = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin1"]

    for encoding in encodings_to_try:
        try:
            with open(schema_path, encoding=encoding) as f:
                schema_data = json.load(f)

                # Compatible with wrapped format: {"data": {"__schema": {...}}}
                if "data" in schema_data and "__schema" in schema_data["data"]:
                    return schema_data["data"]["__schema"]

                # Compatible with official format: {"__schema": {...}}
                if "__schema" in schema_data:
                    return schema_data["__schema"]

                # If we get here, the file was loaded but doesn't have the expected structure
                break
        except UnicodeDecodeError:
            # Try the next encoding
            continue
        except json.JSONDecodeError:
            # Try the next encoding
            continue
        except Exception:
            # Try the next encoding
            continue

    # If we reach here, none of the encodings worked or the schema format was invalid
    msg = (
        "Invalid GraphQL schema format. Expected a JSON file with either '__schema' or 'data.__schema'. "
        "The file may also have an unsupported encoding."
    )
    raise ValueError(
        msg,
    )


def escape_string(s: str) -> str:
    """Escape special characters in a string for use in Python code."""
    if not s:
        return ""

    # Replace backslashes first to avoid double escaping
    s = s.replace("\\", "\\\\")
    # Replace double quotes
    s = s.replace('"', '\\"')
    # Replace single quotes
    s = s.replace("'", "\\'")
    # Replace newlines
    s = s.replace("\n", "\\n")
    # Replace tabs
    s = s.replace("\t", "\\t")
    # Replace carriage returns
    return s.replace("\r", "\\r")



def capitalize_first(s: str) -> str:
    """Capitalize the first letter of a string."""
    if not s:
        return s
    return s[0].upper() + s[1:]


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(capitalize_first(x) for x in components[1:])


def camel_to_snake(camel_str: str) -> str:
    """Convert camelCase to snake_case."""
    # Special handling for names starting with underscore
    if camel_str.startswith("_"):
        # Keep the underscore prefix and handle the rest normally
        remaining = camel_str[1:]
        # If the remaining part is camelCase, convert it
        if remaining and remaining[0].isupper():
            remaining = remaining[0].lower() + remaining[1:]
        snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", remaining).lower()
        return f"db_{snake_case}"  # Prefixed with db_ to indicate database field

    # Handle special case if the string starts with uppercase
    if camel_str and camel_str[0].isupper():
        camel_str = camel_str[0].lower() + camel_str[1:]

    # Convert camelCase to snake_case
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


def extract_type_info(type_data: dict[str, Any], nested_level=0) -> tuple[str, bool]:
    """Recursively extract type information from GraphQL type data.
    Returns a tuple of (type_name, is_required).
    """
    # Handle string type references that we already quoted in an earlier stage
    if isinstance(type_data, str):
        return type_data, False

    kind = type_data.get("kind")

    # Handle non-null types
    if kind == "NON_NULL":
        inner_type, _ = extract_type_info(type_data["ofType"], nested_level + 1)
        return inner_type, True

    # Handle list types
    if kind == "LIST":
        inner_type, inner_required = extract_type_info(type_data["ofType"], nested_level + 1)
        # Wrap the inner type in List
        if inner_required:
            return f"List[{inner_type}]", False
        return f"List[Optional[{inner_type}]]", False

    # Handle scalar types
    if kind == "SCALAR":
        scalar_name = type_data.get("name")
        python_type = map_scalar_type(scalar_name)
        return python_type, False

    # Handle enum, object, interface, union, and input_object types
    if kind in ["ENUM", "OBJECT", "INTERFACE", "UNION", "INPUT_OBJECT"]:
        # Use string reference for user-defined types to avoid order-of-definition issues
        return f"'{type_data.get('name')}'", False

    # Default case
    return "Any", False


def map_scalar_type(scalar_name: str) -> str:
    """Map GraphQL scalar types to Python types."""
    mapping = {
        "String": "str",
        "Int": "int",
        "Float": "float",
        "Boolean": "bool",
        "ID": "str",
    }
    return mapping.get(scalar_name, "Any")


def collect_types(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collect all types from the schema with their fields and dependencies."""
    types = {}
    # Get all types from the schema
    schema_types = schema.get("types", [])

    # First pass: collect basic type info
    for type_data in schema_types:
        name = type_data.get("name")

        # Skip introspection types
        if name.startswith("__"):
            continue

        # Skip schema type
        if name == "Schema":
            continue

        kind = type_data.get("kind")

        type_info = {
            "kind": kind,
            "description": type_data.get("description", ""),
            "fields": [],
            "dependencies": set(),
            "interfaces": [],
            "enum_values": [],
            "input_fields": [],
            "possible_types": [],
        }

        # Process fields for objects and interfaces
        if kind in ["OBJECT", "INTERFACE"] and "fields" in type_data:
            for field in type_data.get("fields", []):
                field_type, is_required = extract_type_info(field.get("type", {}))

                # Create field info
                field_info = {
                    "name": field.get("name"),
                    "snake_name": camel_to_snake(field.get("name")),
                    "description": field.get("description", ""),
                    "type": field_type,
                    "is_required": is_required,
                    "args": field.get("args", []),
                    "is_deprecated": field.get("isDeprecated", False),
                    "deprecation_reason": field.get("deprecationReason"),
                }

                type_info["fields"].append(field_info)

                # Add dependencies
                if not field_type.startswith(("str", "int", "float", "bool", "Any", "List")):
                    type_info["dependencies"].add(field_type)
                elif field_type.startswith("List[") and not any(
                    t in field_type for t in ["str", "int", "float", "bool", "Any"]
                ):
                    # Extract type from List[Type]
                    match = re.search(r"List\[(?:Optional\[)?([^]]+?)(?:\])?]", field_type)
                    if match and match.group(1) not in [
                        "str",
                        "int",
                        "float",
                        "bool",
                        "Any",
                    ]:
                        type_info["dependencies"].add(match.group(1))

        # Process enum values
        if kind == "ENUM" and "enumValues" in type_data:
            for enum_value in type_data.get("enumValues", []):
                enum_info = {
                    "name": enum_value.get("name"),
                    "description": enum_value.get("description", ""),
                    "is_deprecated": enum_value.get("isDeprecated", False),
                    "deprecation_reason": enum_value.get("deprecationReason"),
                }
                type_info["enum_values"].append(enum_info)

        # Process input fields
        if kind == "INPUT_OBJECT" and "inputFields" in type_data:
            for input_field in type_data.get("inputFields", []):
                field_type, is_required = extract_type_info(input_field.get("type", {}))

                input_field_info = {
                    "name": input_field.get("name"),
                    "snake_name": camel_to_snake(input_field.get("name")),
                    "description": input_field.get("description", ""),
                    "type": field_type,
                    "is_required": is_required,
                    "default_value": input_field.get("defaultValue"),
                }

                type_info["input_fields"].append(input_field_info)

                # Add dependencies
                if not field_type.startswith(("str", "int", "float", "bool", "Any", "List")):
                    type_info["dependencies"].add(field_type)
                elif field_type.startswith("List[") and not any(
                    t in field_type for t in ["str", "int", "float", "bool", "Any"]
                ):
                    # Extract type from List[Type]
                    match = re.search(r"List\[(?:Optional\[)?([^]]+?)(?:\])?]", field_type)
                    if match and match.group(1) not in [
                        "str",
                        "int",
                        "float",
                        "bool",
                        "Any",
                    ]:
                        type_info["dependencies"].add(match.group(1))

        # Process interfaces
        if "interfaces" in type_data:
            for interface in type_data.get("interfaces", []) or []:
                interface_name, _ = extract_type_info(interface)
                type_info["interfaces"].append(interface_name)
                type_info["dependencies"].add(interface_name)

        # Process possible types for interfaces and unions
        if "possibleTypes" in type_data:
            for possible_type in type_data.get("possibleTypes", []) or []:
                type_name, _ = extract_type_info(possible_type)
                type_info["possible_types"].append(type_name)
                type_info["dependencies"].add(type_name)

        types[name] = type_info

    return types


def topological_sort(graph: dict[str, dict[str, Any]]) -> list[str]:
    """Sort types topologically so that dependencies come first.
    Uses Kahn's algorithm.
    """
    # Build an adjacency list representation
    adjacency_list = {node: list(data["dependencies"]) for node, data in graph.items()}

    # Calculate in-degrees (how many types depend on this type)
    in_degree = dict.fromkeys(adjacency_list, 0)
    for node, deps in adjacency_list.items():
        for dep in deps:
            if dep in in_degree:  # Ensure the dep exists in our graph
                in_degree[dep] += 1

    # Start with nodes that have no dependencies
    queue = [node for node, degree in in_degree.items() if degree == 0]
    result = []

    # Process the queue
    while queue:
        node = queue.pop(0)
        result.append(node)

        # For each node that depends on this one
        for neighbor in adjacency_list[node]:
            if neighbor in in_degree:  # Ensure the neighbor exists
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    # Check for cycles
    if len(result) != len(graph):
        # There's a cycle, but we'll still return a partial sort
        # Add remaining nodes to result
        for node in graph:
            if node not in result:
                result.append(node)

    return result


def generate_pydantic_models(
    schema_types: dict[str, dict[str, Any]],
    sorted_types: list[str],
) -> str:
    """Generate Pydantic models from the collected schema types."""
    code = "from typing import List, Optional, Any, Dict, Union, ForwardRef, Callable, TypeVar, Generic, Type, cast\n"
    code += "from pydantic import BaseModel, Field\n"
    code += "import sys\n"
    code += "import json\n"
    code += "import httpx\n"
    code += "from enum import Enum\n"
    code += "from graphql_client import GraphQLClient, camel_case\n\n"

    # Add TypeVar for GraphQL client
    code += "# Type variable for generic models\n"
    code += "T = TypeVar('T')\n\n"

    # Add forward references for circular dependencies
    code += "# Forward references for circular dependencies\n"
    for type_name in sorted_types:
        type_info = schema_types[type_name]
        if type_info["kind"] in ["OBJECT", "INPUT_OBJECT", "INTERFACE"]:
            code += f"{type_name}Ref = ForwardRef('{type_name}')\n"
    code += "\n"

    # Generate enum models
    for type_name in sorted_types:
        type_info = schema_types[type_name]
        if type_info["kind"] == "ENUM":
            if type_info["description"]:
                escaped_desc = escape_string(type_info["description"])
                code += f'"""{escaped_desc}"""\n'

            code += f"class {type_name}(str, Enum):\n"

            if not type_info["enum_values"]:
                code += "    pass\n\n"
                continue

            for enum_value in type_info["enum_values"]:
                if enum_value["description"]:
                    escaped_desc = escape_string(enum_value["description"])
                    code += f'    {enum_value["name"]} = "{enum_value["name"]}"  # {escaped_desc}\n'
                else:
                    code += f'    {enum_value["name"]} = "{enum_value["name"]}"\n'

            code += "\n"

    # Generate interface models
    for type_name in sorted_types:
        type_info = schema_types[type_name]
        if type_info["kind"] == "INTERFACE":
            if type_info["description"]:
                escaped_desc = escape_string(type_info["description"])
                code += f'"""{escaped_desc}"""\n'

            code += f"class {type_name}(BaseModel):\n"

            if not type_info["fields"]:
                code += "    pass\n\n"
                continue

            for field in type_info["fields"]:
                # Determine field type string
                field_type = field["type"]
                if not field["is_required"]:
                    if not field_type.startswith("Optional["):
                        field_type = f"Optional[{field_type}]"

                # Prepare field metadata
                field_metadata = []
                if field["name"] != field["snake_name"] or field["name"].startswith("_"):
                    field_metadata.append(f'alias="{field["name"]}"')

                if field["description"]:
                    escaped_desc = escape_string(field["description"])
                    field_metadata.append(f'description="{escaped_desc}"')

                # Set default value
                default_value = "..." if field["is_required"] else "None"

                # Generate field definition
                field_def = f"    {field['snake_name']}: {field_type} = Field({default_value}"
                if field_metadata:
                    field_def += ", " + ", ".join(field_metadata)
                field_def += ")\n"

                code += field_def

                # Add docstring if there's a description
                if field["description"]:
                    escaped_desc = escape_string(field["description"])
                    code += f'    """{escaped_desc}"""\n'

            # Add Config class if needed
            has_aliases = any(
                field["name"] != field["snake_name"] or field["name"].startswith("_")
                for field in type_info["fields"]
            )
            if has_aliases:
                code += "\n    class Config:\n"
                code += "        populate_by_name = True\n"

            code += "\n"

    # Generate input object models
    for type_name in sorted_types:
        type_info = schema_types[type_name]
        if type_info["kind"] == "INPUT_OBJECT":
            if type_info["description"]:
                escaped_desc = escape_string(type_info["description"])
                code += f'"""{escaped_desc}"""\n'

            code += f"class {type_name}(BaseModel):\n"

            if not type_info["input_fields"]:
                code += "    pass\n\n"
                continue

            for field in type_info["input_fields"]:
                # Determine field type string
                field_type = field["type"]
                if not field["is_required"]:
                    if not field_type.startswith("Optional["):
                        field_type = f"Optional[{field_type}]"

                # Prepare field metadata
                field_metadata = []
                if field["name"] != field["snake_name"] or field["name"].startswith("_"):
                    field_metadata.append(f'alias="{field["name"]}"')

                if field["description"]:
                    escaped_desc = escape_string(field["description"])
                    field_metadata.append(f'description="{escaped_desc}"')

                # Set default value
                default_value = "..." if field["is_required"] else "None"

                # Generate field definition
                field_def = f"    {field['snake_name']}: {field_type} = Field({default_value}"
                if field_metadata:
                    field_def += ", " + ", ".join(field_metadata)
                field_def += ")\n"

                code += field_def

                # Add docstring if there's a description
                if field["description"]:
                    escaped_desc = escape_string(field["description"])
                    code += f'    """{escaped_desc}"""\n'

            # Add Config class if needed
            has_aliases = any(
                (field["name"] != field["snake_name"] or field["name"].startswith("_"))
                for field in type_info["input_fields"]
            )
            if has_aliases:
                code += "\n    class Config:\n"
                code += "        populate_by_name = True\n"

            code += "\n"

    # First, generate argument classes for Query/Mutation/Subscription methods
    for type_name in sorted_types:
        type_info = schema_types[type_name]
        if type_info["kind"] == "OBJECT" and type_name in ["Query", "Mutation", "Subscription"]:
            if not type_info["fields"]:
                continue

            for field in type_info["fields"]:
                if not field.get("args"):
                    continue

                field_name = field["name"]
                snake_name = field["snake_name"]
                camel_name = snake_to_camel(snake_name)
                args_class_name = f"{camel_name}Args"

                # Add class description if needed
                if field["description"]:
                    escaped_desc = escape_string(field["description"])
                    code += f'"""{escaped_desc} - Arguments"""\n'

                # Define the args class
                code += f"class {args_class_name}(BaseModel):\n"

                if not field["args"]:
                    code += "    pass\n\n"
                    continue

                # Add fields for each argument
                for arg in field["args"]:
                    arg_type, arg_required = extract_type_info(arg.get("type", {}))
                    arg_name = arg["name"]
                    arg_snake_name = camel_to_snake(arg_name)

                    # Prepare field metadata
                    arg_metadata = []
                    if arg_name != arg_snake_name or arg_name.startswith("_"):
                        arg_metadata.append(f'alias="{arg_name}"')

                    if arg.get("description"):
                        escaped_arg_desc = escape_string(arg["description"])
                        arg_metadata.append(f'description="{escaped_arg_desc}"')

                    # Set default value
                    default_value = "..." if arg_required else "None"

                    # Generate field definition
                    arg_def = f"    {arg_snake_name}: {arg_type}"
                    if not arg_required:
                        if not arg_type.startswith("Optional["):
                            arg_def = f"    {arg_snake_name}: Optional[{arg_type}]"

                    arg_def += f" = Field({default_value}"
                    if arg_metadata:
                        arg_def += ", " + ", ".join(arg_metadata)
                    arg_def += ")\n"

                    code += arg_def

                    # Add docstring if there's a description
                    if arg.get("description"):
                        escaped_arg_desc = escape_string(arg["description"])
                        code += f'    """{escaped_arg_desc}"""\n'

                # Add Config class if needed
                has_aliases = any(
                    (arg["name"] != camel_to_snake(arg["name"]) or arg["name"].startswith("_"))
                    for arg in field["args"]
                )
                if has_aliases:
                    code += "\n    class Config:\n"
                    code += "        populate_by_name = True\n"

                code += "\n"

    # Generate object models
    for type_name in sorted_types:
        type_info = schema_types[type_name]
        if type_info["kind"] == "OBJECT":
            if type_info["description"]:
                escaped_desc = escape_string(type_info["description"])
                code += f'"""{escaped_desc}"""\n'

            # Determine base classes
            base_classes = ["BaseModel"]
            if type_info["interfaces"]:
                base_classes.extend(type_info["interfaces"])

            # Make Query and Mutation inherit from GraphQLClient
            if type_name in ["Query", "Mutation"]:
                base_classes = ["GraphQLClient", "BaseModel"]

            code += f"class {type_name}({', '.join(base_classes)}):\n"

            if not type_info["fields"]:
                code += "    pass\n\n"
                continue

            # Add url field for Query and Mutation classes
            if type_name in ["Query", "Mutation"]:
                code += "    # Definir url como un campo de Pydantic para que no cause error al asignarlo\n"
                code += "    url: Optional[str] = Field(default=None, exclude=True)\n\n"

            # Add constructor for Query and Mutation
            if type_name in ["Query", "Mutation"]:
                code += "    def __init__(self, url: str = None, **data):\n"
                code += "        GraphQLClient.__init__(self, url)\n"
                code += "        BaseModel.__init__(self, **data)\n\n"
                code += "    # GraphQL query/mutation methods\n"

            for field in type_info["fields"]:
                field_name = field["name"]
                field_type, is_required = extract_type_info(field["type"])
                snake_name = field["snake_name"]
                camel_name = snake_to_camel(snake_name)

                # Use args class if arguments exist
                if field.get("args"):
                    args_class_name = f"{camel_name}Args"
                    args_str = f"self, args: {args_class_name}"
                else:
                    args_str = "self"

                # Document the method with implementation
                code += f"    def {snake_name}({args_str}) -> {field_type}:\n"
                if field["description"]:
                    escaped_desc = escape_string(field["description"])
                    code += f'        """{escaped_desc}"""\n'
                if field["is_deprecated"]:
                    deprecation_msg = field["deprecation_reason"] or "No reason provided"
                    # Split the deprecation message by newlines and add # to each line
                    if "\n" in deprecation_msg:
                        lines = deprecation_msg.split("\n")
                        code += f"        # Deprecated: {lines[0]}\n"
                        for line in lines[1:]:
                            code += f"        # {line}\n"
                    else:
                        code += f"        # Deprecated: {deprecation_msg}\n"

                # Generate implementation based on type
                if type_name == "Query":
                    # For queries, call execute_query with appropriate fields
                    code += "        # Determine fields to request based on return type\n"
                    code += f"        fields = self.get_model_fields({field_type})\n"
                    code += "        # Use model_dump for Pydantic v2\n"
                    code += (
                        "        variables = args.model_dump(by_alias=True) if args else None\n"
                    )
                    code += f'        return self.execute_query({field_type}, "{field_name}", fields, variables)\n\n'
                elif type_name == "Mutation":
                    # For mutations, call execute_mutation
                    code += "        # Determine fields to request based on return type\n"
                    code += f"        fields = self.get_model_fields({field_type})\n"
                    code += "        # Use model_dump for Pydantic v2\n"
                    code += (
                        "        variables = args.model_dump(by_alias=True) if args else None\n"
                    )
                    code += f'        return self.execute_mutation({field_type}, "{field_name}", fields, variables)\n\n'
                else:
                    # Subscription not implemented yet
                    code += (
                        '        raise NotImplementedError("Subscriptions not yet implemented")\n\n'
                    )

            # Skip regular field processing
            continue
        # For Subscription, just use stub implementation
        code += "    # GraphQL subscription methods\n"

        for field in type_info["fields"]:
            field_name = field["name"]
            field_type, is_required = extract_type_info(field["type"])
            snake_name = field["snake_name"]
            camel_name = snake_to_camel(snake_name)

            # Use args class if arguments exist
            if field.get("args"):
                args_class_name = f"{camel_name}Args"
                args_str = f"self, args: {args_class_name}"
            else:
                args_str = "self"

            # Document the method with a stub
            code += f"    def {snake_name}({args_str}) -> {field_type}:\n"
            if field["description"]:
                escaped_desc = escape_string(field["description"])
                code += f'        """{escaped_desc}"""\n'
            if field["is_deprecated"]:
                deprecation_msg = field["deprecation_reason"] or "No reason provided"
                # Split the deprecation message by newlines and add # to each line
                if "\n" in deprecation_msg:
                    lines = deprecation_msg.split("\n")
                    code += f"        # Deprecated: {lines[0]}\n"
                    for line in lines[1:]:
                        code += f"        # {line}\n"
                else:
                    code += f"        # Deprecated: {deprecation_msg}\n"
            code += '        raise NotImplementedError("Subscriptions not yet implemented")\n\n'

        # Skip regular field processing
        continue

        # Regular object processing for non-method fields
        for field in type_info["fields"]:
            # Skip fields that have arguments (those are methods, not properties)
            if field["args"]:
                continue

            # Determine field type string
            field_type = field["type"]
            if not field["is_required"] and not field_type.startswith("Optional["):
                field_type = f"Optional[{field_type}]"

            # Prepare field metadata
            field_metadata = []
            if field["name"] != field["snake_name"] or field["name"].startswith("_"):
                field_metadata.append(f'alias="{field["name"]}"')

            if field["description"]:
                escaped_desc = escape_string(field["description"])
                field_metadata.append(f'description="{escaped_desc}"')

            # Add deprecation info
            if field["is_deprecated"]:
                field_metadata.append("deprecated=True")
                if field["deprecation_reason"]:
                    escaped_reason = escape_string(field["deprecation_reason"])
                    field_metadata.append(f'deprecation_reason="{escaped_reason}"')

            # Set default value
            default_value = "..." if field["is_required"] else "None"

            # Generate field definition
            field_def = f"    {field['snake_name']}: {field_type} = Field({default_value}"
            if field_metadata:
                field_def += ", " + ", ".join(field_metadata)
            field_def += ")\n"

            code += field_def

            # Add docstring if there's a description
            if field["description"]:
                escaped_desc = escape_string(field["description"])
                code += f'    """{escaped_desc}"""\n'

        # Add Config class if needed
        has_aliases = any(
            (field["name"] != field["snake_name"] or field["name"].startswith("_"))
            for field in type_info["fields"]
            if not field["args"]
        )
        if has_aliases:
            code += "\n    class Config:\n"
            code += "        populate_by_name = True\n"

        code += "\n"

    # Add code to resolve forward references
    code += "# Resolve forward references\n"
    for type_name in sorted_types:
        type_info = schema_types[type_name]
        if type_info["kind"] in ["OBJECT", "INPUT_OBJECT", "INTERFACE"]:
            code += f"{type_name}.update_forward_refs()\n"

    return code


def main(schema_path: str, output_path: str = "pydantic_models.py") -> None:
    """Generate Pydantic models from a GraphQL schema JSON file."""
    schema = load_graphql_schema(schema_path)

    schema_types = collect_types(schema)

    sorted_types = topological_sort(schema_types)

    code = generate_pydantic_models(schema_types, sorted_types)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    schema_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "pydantic_models.py"

    main(schema_path, output_path)
