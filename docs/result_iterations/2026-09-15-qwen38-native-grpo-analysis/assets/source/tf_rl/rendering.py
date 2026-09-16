"""One shared tools-aware rendering path for filtering and rollout."""
import copy
import json
from .reward import strict_json


def decode_messages(value):
    messages = strict_json(value) if isinstance(value, str) else copy.deepcopy(list(value))
    for message in messages:
        # Arrow nullable fields and API null content must not become text 'None'.
        for key in list(message):
            if message[key] is None:
                message.pop(key)
        if 'content' not in message:
            message['content'] = ''
        if not isinstance(message['content'], str):
            raise ValueError('This package supports text-only histories')
        for call in message.get('tool_calls', []):
            args = call['function']['arguments']
            # Qwen templates expect a decoded arguments object, not JSON inside JSON.
            if isinstance(args, str):
                call['function']['arguments'] = strict_json(args)
    return messages


def decode_tools(value):
    return strict_json(value) if isinstance(value, str) else copy.deepcopy(list(value or []))


def prompt_text(tokenizer, messages, tools, template_kwargs=None):
    kwargs = dict(template_kwargs or {})
    forbidden = {'tools', 'tokenize', 'add_generation_prompt', 'return_tensors', 'return_dict'} & kwargs.keys()
    if forbidden:
        raise ValueError('Reserved chat template kwargs: ' + str(sorted(forbidden)))
    text = tokenizer.apply_chat_template(decode_messages(messages), tools=decode_tools(tools),
        tokenize=False, add_generation_prompt=True, **kwargs)
    if not isinstance(text, str):
        raise TypeError('chat template did not return text')
    for tool in decode_tools(tools):
        if tool['function']['name'] not in text:
            raise ValueError('Chat template omitted advertised tool: ' + tool['function']['name'])
    return text


def prompt_tokens(tokenizer, messages, tools, template_kwargs=None):
    text = prompt_text(tokenizer, messages, tools, template_kwargs)
    # Explicit no-special-token encoding prevents BOS/EOS duplication.
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError('Empty rendered prompt')
    return list(ids)
