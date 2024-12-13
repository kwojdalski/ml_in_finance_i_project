"""ml_in_finance_i_project file for ensuring the package is executable
as `ml-in-finance-i-project` and `python -m ml_in_finance_i_project`
"""

import sys
from pathlib import Path
from typing import Any

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main(*args, **kwargs) -> Any:
    package_name = Path(__file__).parent.name
    configure_project(package_name)

    interactive = hasattr(sys, "ps1")
    kwargs["standalone_mode"] = not interactive

    run = find_run_command(package_name)
    return run(*args, **kwargs)


if __name__ == "__main__":
    main()
