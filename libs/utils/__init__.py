from typing import Tuple, List, Optional
import subprocess
import sys
import importlib.metadata
from packaging import version


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


def check_libary_major_version(library_name: str, required_major_version: str):
    '''
    Raise an exception if not satisfied
    '''
    installed_version = importlib.metadata.version(library_name)

    # Parse the version to handle semantic versioning properly
    installed_major_version = version.parse(installed_version).base_version

    if installed_major_version != required_major_version:
        raise ImportError(f"{library_name} version {required_major_version} is required, but {installed_version} is installed.")
