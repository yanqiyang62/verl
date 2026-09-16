"""Reference-agreement reward, NOT independently verified business success.
Supported output: native Qwen XML-like or legacy JSON <tool_call>, and structured JSON.
No fuzzy text reward. No live APIs are called.
"""
import json
import re
import math
from functools import lru_cache
from tf_rl.qwen_tool_parser import ContractError, parse_xml_calls


def strict_json(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('Duplicate JSON key: ' + k)
            out[k] = v
        return out
    def constant(x):
        raise ValueError('Non-finite JSON constant: ' + x)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def final_text(text):
    text = text.strip()
    # A generation may start inside <think>, depending on the model template.
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1].strip()
    elif '<think>' in text:
        raise ValueError('Unfinished thinking block')
    for token in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>'):
        if text.endswith(token):
            text = text[:-len(token)].rstrip()
    return text


def parse_output(text, kind, allow_fence=False, schemas=None):
    text = final_text(text)
    if kind == 'tool_call':
        first = re.search(r'<tool_call>\s*', text)
        if first and text[first.end():].startswith('<'):
            if schemas is None:
                raise ContractError('Native tool parsing requires declared schemas')
            return parse_xml_calls(text, schemas, strict_json)
        matches = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S))
        if not matches:
            raise ValueError('Expected native <tool_call> JSON output')
        if re.sub(r'<tool_call>\s*.*?\s*</tool_call>', '', text, flags=re.S).strip():
            raise ValueError('Unexpected text around tool call')
        calls = []
        for m in matches:
            obj = strict_json(m.group(1))
            if set(obj) != {'name', 'arguments'} or not isinstance(obj['name'], str):
                raise ValueError('Invalid tool envelope')
            args = obj['arguments']
            if isinstance(args, str):
                args = strict_json(args)
            if not isinstance(args, dict):
                raise ValueError('Arguments must be an object')
            calls.append({'name': obj['name'], 'arguments': args})
        return calls
    if kind != 'json':
        raise ValueError('Free text requires a separate semantic judge')
    if allow_fence:
        match = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            text = match.group(1)
    value = strict_json(text)
    if not isinstance(value, (dict, list)):
        raise ValueError('Expected JSON object or array')
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def leaves(value, path=()):
    if isinstance(value, dict) and value:
        out = {}
        for k, v in value.items():
            out.update(leaves(v, path + (('key', k),)))
        return out
    if isinstance(value, list) and value:
        out = {}
        for i, v in enumerate(value):
            out.update(leaves(v, path + (('index', i),)))
        return out
    return {path: canonical(value)}


@lru_cache(maxsize=4096)
def ground_truth_spec(text):
    spec = strict_json(text)
    if not isinstance(spec, dict) or 'expected' not in spec:
        raise ContractError('Missing ground_truth.expected')
    if spec.get('kind') == 'tool_call':
        import jsonschema
        schemas = spec.get('schemas')
        if not isinstance(schemas, dict) or not isinstance(spec['expected'], list) or not spec['expected']:
            raise ContractError('Missing tool schemas or expected calls')
        for name, schema in schemas.items():
            jsonschema.validators.validator_for(schema).check_schema(schema)
            if schema.get('type') != 'object':
                raise ContractError('Missing object schema: ' + name)
            for key, prop in schema.get('properties', {}).items():
                if prop.get('type') not in ('string', 'integer', 'number', 'boolean', 'array', 'object', 'null'):
                    raise ContractError('Missing/unsupported declared type: ' + name + '.' + key)
        for call in spec['expected']:
            if call.get('name') not in schemas or not isinstance(call.get('arguments'), dict):
                raise ContractError('Invalid expected call or missing tool schema')
            jsonschema.validate(call['arguments'], schemas[call['name']])
    return spec


def score_spec(solution_str, spec):
    zero = {'score': 0.0, 'acc': 0.0, 'field_accuracy': 0.0, 'format_valid': 0.0}
    predicted = parse_output(solution_str, spec['kind'], spec.get('allow_fence', False), spec.get('schemas'))
    expected = spec['expected']
    if spec['kind'] == 'tool_call':
        if len(predicted) != len(expected):
            return zero
        for got, want in zip(predicted, expected):
            if got['name'] != want['name']:
                return zero
            # Validate the advertised function schema before assigning partial credit.
            import jsonschema
            jsonschema.validate(got['arguments'], spec['schemas'][got['name']])
            if got['name'] == 'tool_city_recommendation_everyday':
                n = got['arguments'].get('days_count')
                for k, v in got['arguments'].items():
                    if isinstance(v, list) and len(v) != n:
                        return zero
    a, b = leaves(predicted), leaves(expected)
    match = sum(k in b and b[k] == v for k, v in a.items())
    agreement = 2 * match / max(len(a) + len(b), 1)
    exact = canonical(predicted) == canonical(expected)
    # Sparse bonus for complete reference agreement; typed fields and extra keys matter.
    score = 0.8 * agreement + 0.2 * float(exact)
    return {'score': score, 'acc': float(exact), 'field_accuracy': agreement, 'format_valid': 1.0}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    spec = ground_truth_spec(ground_truth)  # Bad dataset contracts fail loudly.
    if spec.get('kind') not in ('json', 'tool_call'):
        raise ValueError('Use rule_train.parquet; free text has no semantic reward in this package')
    try:
        result = score_spec(solution_str, spec)
        assert math.isfinite(result['score'])
        return result
    except (ValueError, TypeError, KeyError, IndexError):
        return {'score': 0.0, 'acc': 0.0, 'field_accuracy': 0.0, 'format_valid': 0.0}
    except Exception as exc:
        import jsonschema
        if isinstance(exc, jsonschema.ValidationError):
            return {'score': 0.0, 'acc': 0.0, 'field_accuracy': 0.0, 'format_valid': 0.0}
        raise
