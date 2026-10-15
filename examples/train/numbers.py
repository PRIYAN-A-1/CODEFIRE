def triangular_sum(limit):
    result = 0
    for value in range(limit + 1):
        result += value
    return result

def count_positive(values):
    return sum(1 for value in values if value > 0)

assert triangular_sum(4) == 10
assert count_positive([-1, 0, 2, 4]) == 2
