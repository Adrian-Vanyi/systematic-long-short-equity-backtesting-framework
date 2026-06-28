"""Project-wide logging configuration.

Usage
-----
At the top of any notebook or top-level script, add:

    from modules import logging_setup
    logging_setup.configure(level="INFO")  # or "DEBUG" / "WARNING" / "ERROR"

Each project module obtains its own logger via `logging.getLogger(__name__)`.
Once `configure(...)` has been called, messages from those loggers are routed
through a single shared handler installed on the `modules` package logger.

Public API
----------
* function `configure(...)`
        configure the project's package-level logger (`modules`).

* function `is_configured()`
        return whether `configure(...)` has been called.

"""

from __future__ import annotations
import logging
import sys


_CONFIGURED = False


def configure(level: str | int = "INFO", stream = None) -> None:
    """Configure the project's package-level logger (`modules`).

    Installs a single `StreamHandler` on the `modules` logger, which all
    `modules.*` loggers inherit from. Propagation to the Python root logger
    is disabled, so third-party libraries (e.g. `requests`, `urllib3`) are
    not affected by this configuration.

    Idempotent: calling more than once replaces the existing handler
    rather than stacking handlers (which would cause duplicate messages).

    Parameters
    ----------
    level
        Logging level. Either a string ("DEBUG", "INFO", "WARNING", "ERROR")
        or an integer constant from the `logging` module (`logging.DEBUG`,
        `logging.INFO`, `logging.WARNING`, `logging.ERROR`).
    stream
        Stream to write to. Defaults to `sys.stderr`. Pass `sys.stdout`
        for notebook output that interleaves with cell output (and avoids
        the red `stderr` styling in Jupyter).
    """
    global _CONFIGURED

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    project_logger = logging.getLogger("modules")
    project_logger.setLevel(level)

    # Remove existing handlers to make the call idempotent.
    for h in list(project_logger.handlers):
        project_logger.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(fmt="[%(levelname)s %(name)s, %(funcName)s:%(lineno)d]  %(message)s")
    )
    project_logger.addHandler(handler)
    project_logger.propagate = False  # don't double-print via the Python root logger

    _CONFIGURED = True


def is_configured() -> bool:
    """Return whether `configure(...)` has been called."""
    return _CONFIGURED