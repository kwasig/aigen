import os
import cProfile
import pstats
import io
import asyncio
from functools import wraps
from utils.logger import logger

# Allow profiling to be toggled via an environment variable. This lets us avoid
# the heavy overhead of cProfile in production where every request was being
# profiled, leading to noticeable latency. Profiling can be enabled by setting
# `ENABLE_PROFILING=1` in the environment.
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "0") in {"1", "true", "True"}


def profile(func):
    """Decorator to profile a function and log the top cumulative results."""

    # If profiling is disabled, simply return the original function to avoid
    # any overhead.
    if not ENABLE_PROFILING:
        return func

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
