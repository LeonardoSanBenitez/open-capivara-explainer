from typing import Tuple, List, Optional
import subprocess
import sys


def exec_command(command: str) -> Tuple[bool, str]:
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, encoding=sys.getfilesystemencoding())
        worked = True
    except subprocess.CalledProcessError as e:
        output = f'{e.returncode} {str(e.output)}'
        worked = False
    return worked, output


def assert_almost_equal(a: float, b: float, threshold: float=0.01) -> None:
    assert abs(a-b)<threshold, f'a={a} and b={b}'


def enforce_float_or_none(values: list) -> List[Optional[float]]:
    # This raises an exception if the value can't be converted to float... is that what we want?
    return [float(value) if value is not None else None for value in values]
# Pseudo automated test
# assert enforce_float_or_none([1, 2, 3]) == [1.0, 2.0, 3.0]
# assert enforce_float_or_none([1, None, 3]) == [1.0, None, 3.0]
# assert enforce_float_or_none([1, None, 3.0]) == [1.0, None, 3.0]
