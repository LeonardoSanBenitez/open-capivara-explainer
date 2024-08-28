from typing import List, Tuple, Optional, Any
import pandas as pd
import os
import json
import re
from pydantic import BaseModel, computed_field
from functools import cached_property

from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel import KernelContext
from libs.utils.logger import get_logger
from libs.utils import check_libary_major_version
from libs.utils.connector_llm import ConnectorLLM


check_libary_major_version('semantic_kernel', '0.5.0')
logger = get_logger('libs.plugin_database')


def _detect_embedding_call(text: str) -> Optional[str]:
    pattern = r"embedded_question\s*\(\s*'([^']+)'\s*\)"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) > 1:
        raise RuntimeError("Multiple embedded_question calls found.")
    return matches[0] if matches else None


def _replace_embedding_call(text: str, embedding: List[float]) -> str:
    pattern = r"embedded_question\s*\(\s*'([^']*)'\s*\)"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) > 1:
        raise RuntimeError("Multiple embedded_question calls found.")
    if matches:
        embedding_str = f"'{str(embedding)}'"
        return re.sub(pattern, embedding_str, text, count=1, flags=re.DOTALL)
    else:
        raise RuntimeError('No matches found')


class CredentialsAWS(BaseModel):
    '''
    If keys is None, authentication should be done with the current identity
    '''
    region_name: str = 'eu-central-1'
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


class PluginDatabaseRDS(BaseModel):
    max_output_rows: int = 100
    max_output_columns: int = 30
    credentials: CredentialsAWS
    llm_embedding: ConnectorLLM

    @computed_field  # type: ignore
    @cached_property
    def client(self) -> Any:  # -> botocore.client.Lambda:
        import boto3  # type: ignore  # noqa
        return boto3.client(
            'lambda',
            **self.credentials.dict()
        )

    def _exec_query(self, query: str) -> Tuple[int, str, Optional[pd.DataFrame]]:
        '''
        Executes a query in the database
        This function should NOT raise any exception, but return a status code and a message
        Side effect: execs query

        @param query: str, SQL query to be executed
        @return status_code: int, HTTP convention
        @return message: str, error or other relevant message
        @return df: pd.DataFrame, data returned by the query
        '''
        # Preprocess query
        topic = _detect_embedding_call(query)
        if topic:
            embedding: List[float] = self.llm_embedding.embedding_one(topic)
            # print('>>>>>>>>>>>> ORIGINAL QUERY', query)
            query = _replace_embedding_call(query, embedding)
            # print('>>>>>>>>>>>> REPLACED QUERY', query)

        # Execute
        response = self.client.invoke(
            FunctionName='researchbot-query-rds-chatbot-272221232549',
            Payload=json.dumps({'sql_query': query}),
        )
        assert 'StatusCode' in response
        if response['StatusCode']//100 != 2:
            assert 'FunctionError' in response
            return response['StatusCode'], response['FunctionError'], None
        assert 'Payload' in response
        query_results = json.loads(response['Payload'].read().decode('utf-8'))
        assert type(query_results) == dict
        if ('errorMessage' in query_results) and ('errorType' in query_results):
            return 400, f"Boto3 error {query_results['errorType']}: {query_results['errorMessage']}", None
        assert 'statusCode' in query_results
        assert query_results['statusCode']//100 == 2
        assert 'body' in query_results
        assert type(query_results['body']) == list
        df = pd.DataFrame(query_results['body'])  # TODO: would it be possible to get the column names?
        return 200, '', df

    def _exec_query_safe(self, query: str) -> Tuple[int, str, pd.DataFrame]:
        '''
        status_code is guranteed to be 2xx, otherwise raises exception.
        dataframe is guranteed to be non-empty, otherwise raises exception.
        dataframe is guranteed to be within max_output_rows and max_output_columns; if the original dataframe is bigger, it is cropped and that is indicated in the message.
        Side effect: execs query
        '''
        status_code, message, df = self._exec_query(query)
        if status_code // 100 != 2:
            raise RuntimeError(f"Error fetching data: {status_code} {message}")
        assert type(df) == pd.DataFrame
        if df.shape[0] == 0:
            raise RuntimeError(f"No data found")
        if df.shape[0] > self.max_output_rows:
            logger.warning(f"Query returned {df.shape[0]} rows")
            message += f"Result was cropped to first {self.max_output_rows} rows, because it returned too much data.\n"
            df = df.head(20)
            # TODO: modify status code?
        if df.shape[1] > self.max_output_columns:
            logger.warning(f"Query returned {df.shape[1]} columns")
            message += f"Result was cropped to first {self.max_output_columns} columns, because it returned too much data.\n"
            df = df.iloc[:, :20]
        return status_code, message, df

    @kernel_function(
        description="Execute an arbitrary SQL query in RDS."
                    " If you need to execute multiple queries to understand the data, call this function multiple times."
                    " The user rarily knows the exact column names, so be sure to follow the provided schema.",
        name="execute_query",
    )
    @kernel_function_context_parameter(
        name="query",
        description="Exact SQL query to be executed. Use valid SQL RDS syntax."
                    " Run only select queries, don't create or update anything."
                    " Be sure to limit your query to at most 100 rows."
                    " Never select all columns, always filter just what you need."
                    #
                    " There is only one stored procedure you are allow to call, embedded_question(text: str),"
                    " you can use it when searching for items that are similar to an arbitrary string, and it should be used only for ORDER BY statements"
                    " using a distante comparisons (operator <->) with the embedding column;"
                    " Also, you should use this function at most once in a query;"
                    " example usage: ORDER BY embedding <-> embedded_question('renewable energies')",

        type = "string",
        required = True,
    )
    def execute_query(self, context: KernelContext) -> str:
        # Parameter validation
        if 'query' not in context.variables:
            raise ValueError('Missing parameter query')
        if (type(context.variables['query']) != str) and (len(context.variables['query']) > 1):
            raise ValueError('Parameter query should be a string')
        query = context.variables['query']

        # Fetch data
        # logger.debug('>>>>>> LLM REQUESTED THE FOLLOWING QUERY:', query)
        _, message, df = self._exec_query_safe(query)

        # Result formatting
        result = ''
        result += f"{message}Query result:\n"
        result += df.to_markdown(index = False)

        return result
