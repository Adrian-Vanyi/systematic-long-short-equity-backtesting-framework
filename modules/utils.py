import time
from functools import wraps


def timeit(func):
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start

            if duration >= 60:
                print(f"{func.__name__} took {duration / 60:.2f} minutes")
            else:
                print(f"{func.__name__} took {duration:.2f} seconds")

    return wrapper