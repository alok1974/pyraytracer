import random
import sys
from contextlib import contextmanager


def random_range(a: float, b: float) -> float:
    if b < a:
        raise RuntimeError('For random range b > a is required!')

    return a + (b - a) * random.random()


@contextmanager
def progress_bar_context(bar_length=50):
    # Setup action: print a newline
    print()
    yield lambda percentage: print_progress_bar(percentage, bar_length)
    # Teardown action: print a newline and flush
    print('\n')
    sys.stdout.flush()


def print_progress_bar(percentage, bar_length=50):
    # Calculate the number of blocks the progress bar should have
    blocks = int((percentage * bar_length) / 100)

    # Build the progress bar
    progress = '█' * blocks
    spaces = ' ' * (bar_length - blocks)

    # Print the progress bar
    print(f'\r{progress}{spaces} {percentage:.2f}%', end='')
    sys.stdout.flush()
