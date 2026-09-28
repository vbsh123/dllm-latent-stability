"""Shared Base continuation and checkpoint-native Instruct formatting."""
def render_prompt(text, config, tokenizer=None):
    mode=config.get('prompt_format','plain_qa')
    if mode=='plain_qa':
        return f'Question: {text.strip()}\nAnswer:'
    if mode=='raw':
        return text
    if mode=='chat':
        if tokenizer is None or not tokenizer.chat_template:
            raise ValueError('chat prompt_format requires the pinned tokenizer chat template')
        return tokenizer.apply_chat_template(
            [{'role':'user','content':text}], tokenize=False, add_generation_prompt=True)
    raise ValueError('prompt_format must be plain_qa, raw or chat')
