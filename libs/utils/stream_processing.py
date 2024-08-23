from typing import Generator


def accumulate_all_generator(generator: Generator) -> str:
    final = ''
    for chunk in generator:
        final += chunk
    return final
