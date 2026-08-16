def cutoff_distance(value, cutoff):
    return value if cutoff is None or value <= cutoff else cutoff + 1


def normalized_distance(distance, maximum):
    return 0.0 if maximum == 0 else distance / maximum
