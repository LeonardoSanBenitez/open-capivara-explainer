from typing import List
import pandas as pd
import tempfile
import random
import csv
import os
from libs.plugin_orchestrator_evaluator import load_eval_dataset, add_inferences, add_evaluation_scores, analyze_results, evaluate_agent
from libs.utils.connector_llm import ChatCompletionMessage, DefinitionOpenaiTool, ChatCompletion

evaluation_questions = pd.DataFrame([
    ['what is the color of the sky', 'blue', 'chitchat', '', 'john doe', 'easy', False, False],
    ['what is the color of the grass', 'green', 'nature', '', 'john doe', 'medium', False, False],
    ['what is the color of the Poaceae Saccharum', 'green', 'nature', '', 'john doe', 'hard', True, True],
], columns=["question", "gold_answer", "area", "focused_elements", "clarification_by", "difficulty", "out_of_scope_for_now", "super_important"])

class MockAgent():
    def __init__(self, **kwargs):
        pass
    def run(self) -> str:
        return random.choice([
            "hi",
            "How are you doing",
        ])
    
class MockLLM():
    def chat_completion(self, messages: List[ChatCompletionMessage], tool_definitions: List[DefinitionOpenaiTool] = []) -> ChatCompletion:
        score = random.choice(['1', '2', '3', '4', '5'])
        return ChatCompletion(**{'choices': [{'message': {'role': 'assistant', 'content': score}, 'finish_reason': 'stop'}]})

def test_load_eval_dataset():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        evaluation_questions.to_csv(temp_file.name, index=False, quoting=csv.QUOTE_ALL)

        # Full
        df = load_eval_dataset(temp_file.name)
        assert df.shape[0] == 2

        # Full including OOS
        df = load_eval_dataset(temp_file.name, remove_out_of_scope=False, super_important_multiplier=1)
        assert df.shape[0] == 3

        # Limited
        df = load_eval_dataset(temp_file.name, row_limit=1)
        assert df.shape[0] == 1

        # Filter area
        df = load_eval_dataset(temp_file.name, filter_areas=['chitchat'])
        assert df.shape[0] == 1

        df = load_eval_dataset(temp_file.name, filter_areas=['chit'])
        assert df.shape[0] == 1

        df = load_eval_dataset(temp_file.name, filter_areas=['nature'], remove_out_of_scope=False, super_important_multiplier=1)
        assert df.shape[0] == 2

        df = load_eval_dataset(temp_file.name, filter_areas=['nature', 'chitchat'], remove_out_of_scope=False, super_important_multiplier=1)
        assert df.shape[0] == 3
    
        # super_important_multiplier
        df = load_eval_dataset(temp_file.name, filter_difficulty=['hard'], remove_out_of_scope=False, super_important_multiplier=1)
        assert df.shape[0] == 1

        df = load_eval_dataset(temp_file.name, filter_difficulty=['hard'], remove_out_of_scope=False, super_important_multiplier=3)
        assert df.shape[0] == 3

        df = load_eval_dataset(temp_file.name, remove_out_of_scope=False, super_important_multiplier=5)
        assert df.shape[0] == 2 + 1*5

def test_add_inferences():
    df2 = add_inferences(evaluation_questions, agent_class=MockAgent)
    assert evaluation_questions.shape[0] == df2.shape[0]
    assert evaluation_questions.shape[1] == df2.shape[1] - 1

def test_add_evaluation_scores():
    df2 = evaluation_questions.copy()
    df2['model_answer'] = 'hi'
    llm = MockLLM()
    df3 = add_evaluation_scores(llm=llm, eval_dataset_with_inference=df2, metric='metric_similarity_score')
    assert df2.shape[0] == df3.shape[0]
    assert df2.shape[1] == df3.shape[1] - 1
    assert len(set(df3['metric_similarity_score'].values) - {1, 2, 3, 4, 5}) == 0

def test_analyze_results():
    df3 = evaluation_questions.copy()
    df3['model_answer'] = 'hi'
    df3['metric_similarity_score'] = 1
    with tempfile.TemporaryDirectory() as temp_dir:
        analyze_results(results=df3, metric='metric_similarity_score', save_dir=temp_dir)

def test_evaluate_agent():
    with tempfile.TemporaryDirectory() as temp_dir:
        evaluation_questions.to_csv(os.path.join(temp_dir, 'evaluation-questions.csv'), index=False, quoting=csv.QUOTE_ALL)
        evaluate_agent(
            llm=MockLLM(),
            evaluation_list=[
                ('evaluation-questions', ['metric_similarity_score', 'metric_coherence_score']),
            ],
            base_dir=temp_dir,
            agent_class=MockAgent,
        )
        assert os.path.exists(os.path.join(temp_dir, 'evaluation-questions'))
        assert os.path.exists(os.path.join(temp_dir, 'evaluation-questions', 'metric_coherence_score'))
        assert os.path.exists(os.path.join(temp_dir, 'evaluation-questions', 'answers.csv'))
