import cProfile
import pstats
import io
import asyncio
from functools import wraps
from utils.logger import logger


def profile(func):
    """Decorator to profile a function and log the top cumulative results."""

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            try:
                return await func(*args, **kwargs)
            finally:
                pr.disable()
                s = io.StringIO()
                pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
                logger.debug(f"Profiling results for {func.__name__}:\n{s.getvalue()}")

        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            try:
                return func(*args, **kwargs)
            finally:
                pr.disable()
                s = io.StringIO()
                pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
                logger.debug(f"Profiling results for {func.__name__}:\n{s.getvalue()}")

        return sync_wrapper
