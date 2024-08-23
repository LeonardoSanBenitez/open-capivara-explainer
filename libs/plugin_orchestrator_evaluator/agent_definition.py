from typing import List, Generator
from abc import abstractmethod


class SkaylinkAgent():
    @abstractmethod
    def __init__(self, model_id: str, RDS_table_name: str, RDS_database_name: str, message_history: List[dict] = []):
        '''
        @param message_history: a list of dictionaries, each dictionary is a message like `{'role': 'user', 'content': 'hi, my name is leonardo'}`
        '''
        pass

    @abstractmethod
    def run(self) -> str:
        '''
        returns the final answer, whatever steps are done in the middle
        '''
        pass

    @abstractmethod
    def run_stream(self) -> Generator[str, None, None]:
        '''
        Basically the same as `run`, but it returns the answer in chunks
        '''
        pass


# Example usage
'''
agent = MockAgent(
    modelId = 'not-needed',
    RDS_table_name = 'not-needed',
    RDS_database_name = 'not-needed',
    message_history = [{
        'role': 'user',
        'content': 'hi',
    }],
)

print(agent.run())

for chunk in run_stream():
    print(chunk)
'''
