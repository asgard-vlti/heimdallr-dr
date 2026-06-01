import numpy as np


def find_longest_run(bool_arr):
    """
    Given a boolean array, find the longest consecutive run of True values and return its length and start/end indices.
    """
    values = np.asarray(bool_arr)

    if len(values) == 0:
        return 0, 0, 0

    max_length = 0
    max_start = 0
    max_end = 0

    current_length = 0
    current_start = 0

    for i, val in enumerate(values):
        if val:
            if current_length == 0:
                current_start = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
                max_end = i - 1
            current_length = 0

    # Check one more time in case the array ends with True
    if current_length > max_length:
        max_length = current_length
        max_start = current_start
        max_end = len(bool_arr) - 1

    return max_length, max_start, max_end
