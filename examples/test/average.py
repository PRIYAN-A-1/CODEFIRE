def running_average(samples):
    total = 0.0
    results = []
    for index, sample in enumerate(samples, 1):
        total += sample
        results.append(total / index)
    return results

assert running_average([2, 4, 6]) == [2.0, 3.0, 4.0]
