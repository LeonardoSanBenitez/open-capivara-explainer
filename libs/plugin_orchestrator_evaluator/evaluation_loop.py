import os
from typing import Dict, List, Type, Tuple, Literal, Any, Optional
from unidecode import unidecode
import csv
import shutil
import pandas as pd
import json
import matplotlib.pyplot as plt

from libs.utils.logger import get_logger
from libs.utils.connector_llm import ConnectorLLM
from libs.plugin_orchestrator_evaluator import render_metric_prompt, parse_conversation
from libs.plugin_orchestrator_evaluator import SkaylinkAgent


logger = get_logger('libs.evaluation')


def pd_duplicate_rows(df: pd.DataFrame, duplication_key: str, n: int) -> pd.DataFrame:
    '''
    If the value of <duplication_key> is non-nan (anything that is filled, except empty strings), then to duplicate that row n times

    No side effects
    '''
    assert n > 1, "Multiplier should be at least 2"
    assert df.shape[0] > 0, "Empty DataFrame"
    mask = (df[duplication_key].notna()) & (df[duplication_key] != '') & (df[duplication_key] != False) & (df[duplication_key] != 'False')  # noqa
    duplicated_df = df[mask].reindex(df[mask].index.repeat(n-1)).reset_index(drop=True)  # -1 because the original row is concatenated in the end
    result_df = pd.concat([df, duplicated_df], ignore_index=True)
    result_df = result_df.sample(frac=1).reset_index(drop=True)  # suffle
    return result_df


def load_eval_dataset(
    dataset_path: str,
    row_limit: Optional[int] = None,
    remove_out_of_scope: bool = True,
    filter_areas: Optional[List[str]] = None,
    filter_difficulty: Optional[List[str]] = None,
    super_important_multiplier: int = 3,
) -> pd.DataFrame:
    '''
    Side effect: none
    Idempotent: yes

    @param filter_areas: list of areas to filter (only areas that start with any of those strings will be included); if None, don't filter
    @param filter_difficulty: list of difficulties to filter (only exact matches will be included); if None, don't filter
    @param super_important_multiplier: how many times each super important question should be repeated
    '''
    # TODO:
    # add filters (like "only area=chitchat" or "difficulty=easy")
    # super_important multiplier (repeat the rows that are super important)
    df = pd.read_csv(dataset_path)
    logger.info(f"Before filtering, evaluation set contains {df.shape[0]} rows")
    if remove_out_of_scope:
        df = df[~df['out_of_scope_for_now']]
    if filter_areas is not None:
        assert len(filter_areas) > 0
        df['area'] = df['area'].str.strip().str.lower().apply(unidecode)
        df = df[df['area'].str.startswith(tuple(filter_areas))]
    if filter_difficulty is not None:
        assert len(filter_difficulty) > 0
        df['difficulty'] = df['difficulty'].str.strip().str.lower().apply(unidecode)
        df = df[df['difficulty'].isin(filter_difficulty)]
    if super_important_multiplier > 1 and df['super_important'].sum()>0:
        logger.info(f"Before expanding super-important, evaluation set contains {df.shape[0]} rows")
        df = pd_duplicate_rows(df.copy(), 'super_important', super_important_multiplier)
        logger.info(f"After expanding super-important, evaluation set contains {df.shape[0]} rows")
    if row_limit:
        assert row_limit > 0
        df = df.head(row_limit)

    assert df.shape[0] > 0
    assert df.shape[1] >= 7
    return df


def add_inferences(eval_dataset: pd.DataFrame, agent_class: Type[SkaylinkAgent]) -> pd.DataFrame:
    '''
    Side effect: calls LLM
    Idempotent: yes (but probabilistic)
    '''
    assert "question" in eval_dataset.columns
    assert "area" in eval_dataset.columns
    assert "focused_elements" in eval_dataset.columns
    assert "clarification_by" in eval_dataset.columns
    assert "difficulty" in eval_dataset.columns
    assert "out_of_scope_for_now" in eval_dataset.columns
    assert "super_important" in eval_dataset.columns

    eval_dataset_with_inference = eval_dataset.copy()
    eval_dataset_with_inference['model_answer'] = ''
    for index, row in eval_dataset_with_inference.iterrows():
        agent = agent_class(
            model_id = 'not-needed',
            RDS_table_name = 'not-needed',
            RDS_database_name = 'not-needed',
            message_history = [{
                'role': 'user',
                'content': row['question'],
            }],
        )
        model_answer = agent.run()
        assert type(model_answer) == str
        assert len(model_answer) > 0
        # TODO: Add graph placeholder
        eval_dataset_with_inference.loc[index, 'model_answer'] = model_answer

    assert eval_dataset.shape[0] == eval_dataset_with_inference.shape[0]
    assert eval_dataset.shape[1] == eval_dataset_with_inference.shape[1] - 1
    return eval_dataset_with_inference


def add_evaluation_scores(
    llm: ConnectorLLM,
    eval_dataset_with_inference: pd.DataFrame,
    metric: Literal['metric_coherence_score', 'metric_protectiveness_score', 'metric_similarity_score'],
) -> pd.DataFrame:
    '''
    Side effect: calls LLM
    Idempotent: yes (but probabilistic)

    Metric values are returned in column <metric>, as integers
    '''
    assert "question" in eval_dataset_with_inference.columns
    assert "area" in eval_dataset_with_inference.columns
    assert "focused_elements" in eval_dataset_with_inference.columns
    assert "clarification_by" in eval_dataset_with_inference.columns
    assert "difficulty" in eval_dataset_with_inference.columns
    assert "out_of_scope_for_now" in eval_dataset_with_inference.columns
    assert "super_important" in eval_dataset_with_inference.columns
    # assert "gold_answer" in eval_dataset_with_inference.columns  # metric-specific
    assert "model_answer" in eval_dataset_with_inference.columns  # metric-specific

    results = eval_dataset_with_inference.copy()
    results[metric] = ''
    for index, row in results.iterrows():
        # TODO: if for any reason the evaluation fails, we should... retry?
        params = {'question': row['question'], 'model_answer': row['model_answer']}
        if 'gold_answer' in row:
            params['gold_answer'] = row['gold_answer']
        conversation = render_metric_prompt(metric, params)
        parsed_conversation = parse_conversation(conversation)
        chat_completion = llm.chat_completion(parsed_conversation)
        score = chat_completion.choices[0].message.content
        # score = random.choice(['1', '2', '3', '4', '5'])
        results.loc[index, metric] = score

    # Cleaning
    logger.info(f"For metric {metric}, {(results[metric].str.len() != 1).sum()} tests failed")
    results = results[results[metric].str.len() == 1]
    results[metric] = pd.to_numeric(results[metric], errors='coerce')
    results.dropna(subset=['area', 'difficulty', metric], inplace=True)

    assert eval_dataset_with_inference.shape[0] == results.shape[0]
    assert eval_dataset_with_inference.shape[1] == results.shape[1] - 1, f'{eval_dataset_with_inference.shape[1]} vs {results.shape[1]}'
    assert metric in results.columns
    return results


def analyze_results(results: pd.DataFrame, metric: str, save_dir: str) -> None:
    '''
    Side effect: modifies the files at `save_dir`
    Idempotent: yes
    Assumes the folder already exists
    '''
    results = results.copy()
    analysis: Dict[str, Any]= {
        'metric_name': metric,
        'metric_name_pretty': metric.replace('metric_', '').replace('_', ' ').title(),
    }

    # Total
    analysis['count_total'] = int(results.shape[0])
    analysis['mean_total'] = float(results[metric].mean())
    logger.info(f"Mean {analysis['metric_name_pretty']} among all {analysis['count_total']} questions: {analysis['mean_total']}")

    # Super Important
    results_si = results[results['super_important']]
    analysis['count_super_important'] = int(results_si.shape[0])
    analysis['mean_super_important'] = float(results_si[metric].mean())
    logger.info(f"Mean {analysis['metric_name_pretty']} among {analysis['count_super_important']} super-important questions: {analysis['mean_super_important']}")

    # Per area
    results['area'] = results['area'].str.strip().str.lower().apply(unidecode)
    grouped_results = results.groupby('area')[metric].agg(['mean', 'std', 'count'])
    analysis['mean_per_area'] = grouped_results.to_markdown()
    logger.info(f"Results for {analysis['metric_name_pretty']} per area:\n{analysis['mean_per_area']}")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped_results.index, grouped_results['mean'], yerr=grouped_results['std'], capsize=5, color='skyblue')

    for bar in bars:
        yval = bar.get_height()
        # Adding value labels above each bar
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), va='bottom')

    plt.title(f"Mean {analysis['metric_name_pretty']} per area, with std bars")
    plt.xlabel("Area")
    plt.ylabel("Mean Score")
    plt.xticks(rotation=60)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'graph_per_area.png'))
    plt.show()

    # Per difficulty
    results['difficulty'] = results['difficulty'].str.strip().str.lower().apply(unidecode)
    grouped_results = results.groupby('difficulty')[metric].agg(['mean', 'std', 'count'])
    analysis['mean_per_difficulty'] = grouped_results.to_markdown()
    logger.info(f"Results for {analysis['metric_name_pretty']} per difficulty:\n{analysis['mean_per_difficulty']}")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped_results.index, grouped_results['mean'], yerr=grouped_results['std'], capsize=5, color='skyblue')

    for bar in bars:
        yval = bar.get_height()
        # Adding value labels above each bar
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), va='bottom')

    plt.title(f"Mean {analysis['metric_name_pretty']} per difficulty, with std bars")
    plt.xlabel("Difficulty")
    plt.ylabel("Mean Score")
    plt.xticks(rotation=60)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'graph_per_difficulty.png'))
    plt.show()

    # Save
    with open(os.path.join(save_dir, 'analysis.json'), 'w') as f:
        f.write(json.dumps(analysis, indent=4))


def evaluate_agent(
    llm: ConnectorLLM,
    evaluation_list: List[Tuple[str, List[Literal['metric_coherence_score', 'metric_protectiveness_score', 'metric_similarity_score']]]],
    base_dir: str,
    agent_class: Type[SkaylinkAgent],
) -> None:
    '''
    Side effect: modifies the files at `base_dir`, calls LLM
    Idempotent: yes (but probabilistic)
    Assumes `base_dir` contains the datasets specified in evaluation_list, named accordingly
    Does not assumes the folder already exists

    @param evaluation_list: List of (<dataset>, <list of metrics>)
    The evaluation data should contain:
      * all generic columns (question, area, focused_elements, clarification_by, difficulty, out_of_scope_for_now, super_important
      * metric-specific columns (gold_answer, model_answer, and potentially others) for the metrics specified

    @returns none: all results are saved to `base_dir`
    '''
    for dataset, metrics in evaluation_list:
        save_dir = os.path.join(base_dir, dataset)

        # Ensure that the directory is clean
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        eval_dataset = load_eval_dataset(os.path.join(base_dir, dataset+'.csv'))
        eval_dataset_with_inference = add_inferences(eval_dataset, agent_class)
        results = eval_dataset_with_inference.copy()
        for metric in metrics:
            results = add_evaluation_scores(llm, results, metric=metric)
            save_dir_metric = os.path.join(save_dir, metric)
            os.makedirs(save_dir_metric, exist_ok=True)
            analyze_results(results, metric=metric, save_dir=save_dir_metric)

        # The column 'good_answer' is calculated with a fixed threshold over all columns starting with "metric_"
        # TODO: maybe this mechanism should be more flexible?
        results['good_answer'] = results.filter(regex='metric_').apply(lambda x: x > 3).all(axis=1).map({True: 'Yes (automatically inferred)', False: 'No (automatically inferred)'})
        results['issue'] = ''  # to be completed manually
        results['good_answer_transformed'] = ''  # to be calculated after the manual check, with the equation `=TRIM(SUBSTITUTE(LOWER(J1), "(automatically inferred)", ""))`

        results.to_csv(os.path.join(save_dir, 'answers.csv'), index=False, quoting=csv.QUOTE_ALL)
