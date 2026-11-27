from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import socket
import sys

from loguru import logger as _base_logger

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "{extra[ctx]}"
    "<level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {extra[ctx]}{message}"

_CONFIGURED = False


def _format_context(rank: int | None = None, include_host: bool = False) -> str:
    """Build a short optional prefix showing host and peer rank for distributed logs.

    Args:
        rank: Optional integer peer rank, or None for single-process runs.
        include_host: Whether to include the short hostname alongside the rank.

    Returns:
        Formatted context prefix ending with ' | ', or an empty string if unset.
    """
    parts: list[str] = []
    if include_host:
        parts.append(socket.gethostname().split(".")[0])
    if rank is not None:
        parts.append(f"r={rank}")
    return f"{':'.join(parts)} | " if parts else ""


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    quiet: bool = False,
) -> None:
    """Configure loguru stderr formatting and an optional persistent log file sink.

    Args:
        level: Minimum log level for stderr ('DEBUG', 'INFO', 'WARNING', or 'ERROR').
        log_file: Optional file path receiving uncolored timestamped log lines.
        quiet: If True, raise the stderr threshold to WARNING while keeping file logs at level.

    Returns:
        None.
    """
    global _CONFIGURED
    norm_level = level.upper()
    stderr_level = "WARNING" if quiet else norm_level
    use_color = bool(getattr(sys.stderr, "isatty", lambda: False)())

    _base_logger.remove()
    _base_logger.configure(extra={"ctx": ""})
    _base_logger.add(
        sys.stderr,
        level=stderr_level,
        format=_DEFAULT_FORMAT if use_color else _FILE_FORMAT,
        colorize=use_color,
        backtrace=False,
        diagnose=False,
    )
    if log_file is not None:
        dst = Path(log_file)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _base_logger.add(
            dst,
            level=norm_level,
            format=_FILE_FORMAT,
            encoding="utf-8",
            backtrace=True,
            diagnose=False,
        )
    _CONFIGURED = True


def get_logger(rank: int | None = None, include_host: bool = False):
    """Return a loguru logger with optional host and rank context.

    Args:
        rank: Optional peer rank bound into every log line prefix.
        include_host: Whether to prefix messages with the short machine hostname.

    Returns:
        Bound loguru logger instance.
    """
    if not _CONFIGURED:
        configure_logging()
    return _base_logger.bind(ctx=_format_context(rank=rank, include_host=include_host))


@contextmanager
def run_log_sink(log_path: str | Path | None, level: str = "INFO") -> Iterator[None]:
    """Attach a per-run log file sink for the duration of a with-block.

    Args:
        log_path: Target log file path (e.g., output_dir / 'run.log'), or None to disable.
        level: Minimum log level written to the per-run file.

    Returns:
        Context manager yielding None while the file sink is active.
    """
    if log_path is None:
        yield
        return
    dst = Path(log_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sink_id = _base_logger.add(
        dst,
        level=level.upper(),
        format=_FILE_FORMAT,
        encoding="utf-8",
        mode="w",
        backtrace=True,
        diagnose=False,
    )
    try:
        yield
    finally:
        _base_logger.remove(sink_id)
