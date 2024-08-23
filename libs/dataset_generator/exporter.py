
import json
import os
from typing import Tuple, Literal, List, Dict
import tempfile
import pandas as pd
from pydantic import BaseModel, PrivateAttr
from jinja2 import Template
from huggingface_hub import HfApi

from libs.utils.logger import get_logger
from libs.utils.connector_llm import ChatCompletionMessage


logger = get_logger('libs.dataset_generator')


class DatasetExporter(BaseModel):
    dataset: List[List[ChatCompletionMessage]]
    prompt_format: Literal['zephyr', 'chatml', 'llama2', 'llama3'] = 'zephyr'
    dataset_text_field: str = "text"

    # Every template receives the variables:
    # - messages: List[Dict[str, str]] with keys 'role' and 'content'; serialization of ChatCompletionMessage
    # - eos_token: str; end of sentence token
    # - add_generation_prompt: bool; if True, add a generation prompt at the end of the text
    _prompt_format_templates: Dict[str, str] = PrivateAttr(default={
        'zephyr': "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}{% endfor %}",  # noqa: E501
    })

    @staticmethod
    def _split_df(df: pd.DataFrame, test_size: float, validation_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not (0 < test_size < 1) or not (0 < validation_size < 1):
            raise ValueError("test_size and validation_size must be floats between 0 and 1")
        if test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1")

        total_size = df.shape[0]
        test_len = int(total_size * test_size)
        validation_len = int(total_size * validation_size)

        shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = shuffled_df.iloc[:test_len]
        validation_df = shuffled_df.iloc[test_len:test_len + validation_len]
        train_df = shuffled_df.iloc[test_len + validation_len:]

        assert train_df.shape[0] + validation_df.shape[0] + test_df.shape[0] == df.shape[0]
        return train_df, validation_df, test_df

    def _convert_trajectory_to_str(self, trajectory: List[ChatCompletionMessage]):
        template = self._prompt_format_templates[self.prompt_format]
        result = Template(template).render(
            messages=[message.dict() for message in trajectory],
            eos_token="</s>",
            add_generation_prompt=False,
        )
        return result

    def _convert_dataset_to_df(self) -> pd.DataFrame:
        '''
        Convert messages to trainable text
        Side-effect free
        '''
        if self.prompt_format != 'zephyr':
            raise NotImplementedError("Only zephyr format is supported at the moment")
        assert self.prompt_format in self._prompt_format_templates.keys()
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")

        rows: List[str] = []
        for row in self.dataset:
            # TODO: this is not zefhir...
            rows.append(self._convert_trajectory_to_str(row))
        assert all([type(row)==str for row in rows])
        df = pd.DataFrame(rows, columns=[self.dataset_text_field])
        assert df.shape[0] == len(self.dataset)
        assert df.shape[1] == 1
        return df


class DatasetExporterHuggingFace(DatasetExporter):
    repo_id: str
    dataset_config: str

    async def export(self) -> None:
        # TODO: async, maybe using run_as_future
        # https://huggingface.co/docs/huggingface_hub/guides/upload#non-blocking-uploads
        df = self._convert_dataset_to_df()

        # Upload each split to HF
        assert type(os.getenv('HF_TOKEN')) == str
        assert len(os.getenv('HF_TOKEN')) > 0  # type: ignore
        api = HfApi()
        df_train, df_validation, df_test = self._split_df(df, test_size=0.1, validation_size=0.1)

        for df_split, dataset_split in [(df_train, 'train'), (df_validation, 'validation'), (df_test, 'test')]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
                df_split.to_parquet(f.name, engine='pyarrow', index=False)
                api.upload_file(
                    path_or_fileobj=f.name,
                    path_in_repo=os.path.join(self.dataset_config, dataset_split, f"data.parquet"),
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )
            logger.info(f'Uploaded {dataset_split} with {df_split.shape[0]} rows')


# Pseudo tests
# This actually uploads to HF
# Requries the env variable HF_TOKEN to be set
'''
dataset = gen_dataset()
await DatasetExporterHuggingFace(
    dataset=dataset,
    repo_id = "LeonardoBenitez/capivara-plugin-orchestration",
    dataset_config='plugins_capital_v1.0',
).export()
'''


###############################
class DatasetExporterLocal(DatasetExporter):
    path_local_base: str

    async def export(self) -> None:
        '''
        Saves 3 parquet files with the names train.parquet, validation.parquet and test.parquet
        '''
        if not os.path.exists(self.path_local_base):
            os.makedirs(self.path_local_base)

        df = self._convert_dataset_to_df()
        df_train, df_validation, df_test = self._split_df(df, test_size=0.1, validation_size=0.1)
        for df_split, dataset_split in [(df_train, 'train'), (df_validation, 'validation'), (df_test, 'test')]:
            df_split.to_parquet(os.path.join(self.path_local_base, dataset_split + '.parquet'), engine='pyarrow', index=False)


# Pseudo tests
# This actually saves a file
'''
dataset = gen_dataset()
await DatasetExporterLocal(
    dataset=dataset,
    path_local_base = '/src/data/plugins_capital_v1.0',
).export()
'''
