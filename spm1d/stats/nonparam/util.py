
import itertools
import numpy as np


def permutations_without_repetition(labels):
    '''
    Generate all distinct orderings of *labels*.

    Positions are assigned to the smallest label first and then to each
    following label among the positions that remain.
    '''
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    n      = len(labels)
    if n == 0:
        yield []
        return
    values = sorted(set(labels))
    counts = [labels.count(v) for v in values]
    perm   = [None] * n
    free   = [list(range(n))]
    stack  = [itertools.combinations(free[0], counts[0])]
    while stack:
        k      = len(stack) - 1
        chosen = next(stack[k], None)
        if chosen is None:
            stack.pop()
            free.pop()
            continue
        for i in chosen:
            perm[i] = values[k]
        if k == len(values) - 1:
            yield list(perm)
        else:
            free.append( [i for i in free[k] if i not in chosen] )
            stack.append( itertools.combinations(free[k+1], counts[k+1]) )
