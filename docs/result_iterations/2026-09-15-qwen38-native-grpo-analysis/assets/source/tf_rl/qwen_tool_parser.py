# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synchronous reward adapter for Qwen3XMLToolParser's XML-like wire format.

Adapted from verl/experimental/agent_loop/tool_parser.py at
d5d24f87ff26025cec320d620150001f4b0d7a1c (see THIRD_PARTY.md).
Preserves schema-driven conversion and one boundary newline removal. Unlike
the serving parser, rejects truncated calls, duplicate parameters and invalid
typed values. Does not recover partial calls or modify rollout token IDs.
"""
import json
import re


class ToolParseError(ValueError):
    """The generated wire syntax is invalid."""


class ToolSchemaError(ValueError):
    """The generated tool or argument cannot satisfy its declared schema."""


class ContractError(RuntimeError):
    """Dataset configuration error; must never be converted into zero reward."""


IDENTIFIER = r'[A-Za-z0-9_.-]+'
CONTROL = re.compile(r'</?(?:tool_call|function|parameter)(?:[=>\s]|$)')


def parse_xml_structure(text):
    """Parse the entire response, allowing prose only before the first call.

    Raw parameter values retain whitespace except one leading/trailing newline,
    matching the native template and the community parser. This is XML-like,
    not XML: no entity unescaping, attribute parsing, or arbitrary JSON extraction.
    """
    start = text.find('<tool_call>')
    if start < 0:
        raise ToolParseError('missing_tool_call')
    prefix = text[:start]
    if CONTROL.search(prefix) or '```' in prefix:
        raise ToolParseError('invalid_prefix_or_code_fence')
    tail = text[start:]
    calls = []
    while tail:
        match = re.match(r'<tool_call>\s*<function=(' + IDENTIFIER + r')>\s*', tail)
        if not match:
            raise ToolParseError('malformed_call_or_unexpected_suffix')
        name = match[1]
        tail = tail[match.end():]
        arguments = {}
        while tail.startswith('<parameter='):
            param = re.match(r'<parameter=(' + IDENTIFIER + r')>(.*?)</parameter>\s*', tail, re.S)
            if not param:
                raise ToolParseError('malformed_or_truncated_parameter')
            key, value = param[1], param[2]
            if key in arguments:
                raise ToolParseError('duplicate_parameter:' + key)
            if CONTROL.search(value):
                raise ToolParseError('nested_or_unclosed_control_tag')
            if value.startswith('\n'):
                value = value[1:]
            if value.endswith('\n'):
                value = value[:-1]
            arguments[key] = value
            tail = tail[param.end():]
        end = re.match(r'</function>\s*</tool_call>\s*', tail)
        if not end:
            raise ToolParseError('malformed_or_truncated_call_end')
        tail = tail[end.end():]
        calls.append({'name': name, 'arguments': arguments})
    return calls


def convert_xml_calls(calls, schemas, loads):
    """Convert only by declared types, never by teacher argument values."""
    converted = []
    for call in calls:
        name = call['name']
        if name not in schemas:
            raise ToolSchemaError('undeclared_tool:' + name)
        props = schemas[name].get('properties', {})
        arguments = {}
        for key, raw in call['arguments'].items():
            if key not in props:
                raise ToolSchemaError('undeclared_parameter:' + name + '.' + key)
            typ = props[key].get('type')
            if typ == 'string':
                value = raw  # Literal "null", spaces and newlines remain strings.
            elif typ in ('integer', 'number', 'boolean', 'array', 'object', 'null'):
                try:
                    value = loads(raw)
                    json.dumps(value, allow_nan=False)
                except ValueError as exc:
                    raise ToolSchemaError('invalid_typed_value:' + key) from exc
                valid = {
                    'integer': type(value) is int,
                    'number': type(value) in (int, float),
                    'boolean': type(value) is bool,
                    'array': isinstance(value, list),
                    'object': isinstance(value, dict),
                    'null': value is None,
                }[typ]
                if not valid:
                    raise ToolSchemaError('wrong_parameter_type:' + key)
            else:
                raise ContractError('Missing/unsupported declared type: ' + name + '.' + key)
            arguments[key] = value
        converted.append({'name': name, 'arguments': arguments})
    return converted


def parse_xml_calls(text, schemas, loads):
    return convert_xml_calls(parse_xml_structure(text), schemas, loads)
