from libs.plugin_orchestrator_evaluator import render_metric_prompt, parse_conversation
from libs.utils.connector_llm import ChatCompletionMessage


def test_render_metric_prompt():
    metric = 'metric_coherence_score'
    prompt = render_metric_prompt(metric, {'question': 'hi', 'model_answer': 'hello'})
    assert type(prompt) == str
    assert len(prompt) > 10
    assert 'hello' in prompt

    metric = 'metric_protectiveness_score'
    prompt = render_metric_prompt(metric, {'question': 'hi', 'model_answer': 'hello'})
    assert type(prompt) == str
    assert len(prompt) > 10
    assert 'hello' in prompt

    metric = 'metric_similarity_score'
    prompt = render_metric_prompt(metric, {'question': 'hi', 'gold_answer': 'hi', 'model_answer': 'hello'})
    assert type(prompt) == str
    assert len(prompt) > 10
    assert 'hello' in prompt

def test_parse_conversation():
    conversation = '''
    User:
    question: What can you tell me about climate change and its effects on the environment?
    answer: Climate change has far-reaching effects on the environment. Rising temperatures result in the melting of polar ice caps, contributing to sea-level rise. Additionally, more frequent and severe weather events, such as hurricanes and heatwaves, can cause disruption to ecosystems and human societies alike.

    Assistant:
    5

    User:
    question: What is the color of the sky?
    answer: The sky is blue.

    Assistant:
    5
    '''
    parsed_conversation = parse_conversation(conversation)
    assert len(parsed_conversation) == 4
    assert all([type(m) == ChatCompletionMessage for m in parsed_conversation])