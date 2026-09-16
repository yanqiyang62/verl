python - <<'PY'
from transformers import AutoTokenizer

MODEL = "/shared/models/Qwen3.8-27B"

tok = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

print("tokenizer:", tok.name_or_path)
print("=" * 80)

template = tok.chat_template or ""

print("template length:", len(template))
print("has reasoning_effort:", "reasoning_effort" in template)
print("has preserve_thinking:", "preserve_thinking" in template)
print("has tool_call:", "<tool_call>" in template)

print("\n===== rendered prompt =====")

messages = [
    {"role": "user", "content": "你好"}
]

print(
    tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
)
PY
