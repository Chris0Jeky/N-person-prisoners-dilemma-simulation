"""Dependency-free validation of scenario/config files (W5).

The schemas ship inside the package as data (``npdl/experiments/schemas/``), so
validation also works from an installed wheel. The repo-level
``scenarios/schema.json`` and ``configs/schema.json`` are the published,
documented copies; a test keeps them byte-identical to the packaged ones. The
schemas are standard JSON Schema (draft 2020-12); this module implements the
subset they use so validation needs no third-party package:

``$ref`` (local pointers), ``anyOf``, ``type``, ``required``,
``properties``, ``additionalProperties``, ``items``, ``minItems``,
``minProperties``, ``minLength``, ``minimum``, ``maximum``, ``enum``.
"""

import json
import os
from typing import Any, Dict, List

SCENARIO_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "schemas", "scenario.schema.json"
)
CONFIG_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "schemas", "config.schema.json"
)


class ValidationError(Exception):
    """Raised when a file fails schema validation."""

    def __init__(self, path: str, errors: List[str]) -> None:
        self.path = path
        self.errors = errors
        super().__init__(
            f"{path}: {len(errors)} validation error(s):\n" + "\n".join(errors)
        )


def _resolve_ref(ref: str, root: Dict[str, Any]) -> Dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"only local refs supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return _type_name(value) == expected


def _join_path(base: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{base}[{key}]" if base else f"[{key}]"
    return f"{base}.{key}" if base else str(key)


def _validate(
    value: Any, schema: Dict[str, Any], root: Dict[str, Any], path: str
) -> List[str]:
    if "$ref" in schema:
        return _validate(value, _resolve_ref(schema["$ref"], root), root, path)

    errors: List[str] = []
    label = path or "<root>"

    expected_type = schema.get("type")
    if expected_type is not None:
        expected = [expected_type] if isinstance(expected_type, str) else expected_type
        if not any(_matches_type(value, name) for name in expected):
            errors.append(f"{label}: expected {expected_type}, got {_type_name(value)}")
            return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label}: {value!r} is not one of {schema['enum']!r}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{_join_path(path, key)}: missing required property")
        properties = schema.get("properties", {})
        for key, sub in value.items():
            if key in properties:
                errors.extend(
                    _validate(sub, properties[key], root, _join_path(path, key))
                )
            else:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    errors.append(
                        f"{_join_path(path, key)}: additional property not allowed"
                    )
                elif isinstance(extra, dict):
                    errors.extend(_validate(sub, extra, root, _join_path(path, key)))
        min_props = schema.get("minProperties")
        if min_props is not None and len(value) < min_props:
            errors.append(f"{label}: fewer than {min_props} properties")

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, sub in enumerate(value):
                errors.extend(_validate(sub, items, root, _join_path(path, index)))
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{label}: fewer than {min_items} items")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < min_length:
            errors.append(f"{label}: shorter than {min_length} characters")

    if _matches_type(value, "number"):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            errors.append(f"{label}: {value} is less than minimum {minimum}")
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            errors.append(f"{label}: {value} is greater than maximum {maximum}")

    if "anyOf" in schema:
        branches = []
        for option in schema["anyOf"]:
            option_errors = _validate(value, option, root, path)
            if not option_errors:
                break
            branches.append(option_errors)
        else:
            errors.append(f"{label}: does not match any allowed variant")
            for option_errors in branches:
                errors.extend(f"  variant: {message}" for message in option_errors[:3])

    return errors


def validate(data: Any, schema: Dict[str, Any]) -> List[str]:
    """Validate ``data`` against a schema dict; return error strings (empty = valid)."""
    return _validate(data, schema, schema, "")


def _load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path) as f:
        return json.load(f)


def validate_file(path: str, schema_path: str) -> List[str]:
    """Validate a JSON file against a schema file; return errors (empty = valid)."""
    with open(path) as f:
        data = json.load(f)
    return validate(data, _load_schema(schema_path))


def validate_scenario_file(path: str) -> None:
    """Validate a ``scenarios/*.json`` file; raise :class:`ValidationError` if invalid."""
    errors = validate_file(path, SCENARIO_SCHEMA_PATH)
    if errors:
        raise ValidationError(path, errors)


def validate_config_file(path: str) -> None:
    """Validate a ``configs/*.json`` file; raise :class:`ValidationError` if invalid."""
    errors = validate_file(path, CONFIG_SCHEMA_PATH)
    if errors:
        raise ValidationError(path, errors)
