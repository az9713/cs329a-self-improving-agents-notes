"""Deterministic checks of teaching examples; no model calls or network access."""
import json
import math
import statistics
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

def coverage(p, k):
    if not 0 <= p <= 1 or not isinstance(k, int) or k < 0:
        raise ValueError('p must be a probability and k a nonnegative integer')
    return 1 - (1-p)**k

def allocate_samples(probabilities, budget):
    """Maximize separable oracle coverage for known p and equal costs."""
    if not isinstance(budget, int) or budget < 0:
        raise ValueError('budget must be a nonnegative integer')
    for p in probabilities:
        coverage(p, 0)
    if not probabilities and budget:
        raise ValueError('cannot allocate to an empty task collection')
    counts = [0] * len(probabilities)
    for _ in range(budget):
        best = max(range(len(counts)), key=lambda i:
                   probabilities[i] * (1 - probabilities[i]) ** counts[i])
        counts[best] += 1
    return counts

def pass_at_k(n, c, k):
    if not all(isinstance(v, int) for v in (n, c, k)) or not 0 <= c <= n or not 0 <= k <= n:
        raise ValueError('Require integers with 0 <= c <= n and 0 <= k <= n')
    return 1 - (math.comb(n-c, k) if n-c >= k else 0) / math.comb(n, k)

def acceptance_precision(p, tpr, fpr):
    if not all(0 <= v <= 1 for v in (p, tpr, fpr)):
        raise ValueError('Inputs must be probabilities')
    denominator = p*tpr + (1-p)*fpr
    return p*tpr/denominator if denominator else None

def horizon(h50, beta, reliability):
    if not all(math.isfinite(v) for v in (h50, beta, reliability)):
        raise ValueError('inputs must be finite')
    if h50 <= 0 or beta <= 0 or not 0 < reliability < 1:
        raise ValueError('positive duration and slope; reliability in (0, 1)')
    return h50 * math.exp(-math.log(reliability/(1-reliability))/beta)

def group_advantages(rewards):
    if not rewards or not all(math.isfinite(v) for v in rewards):
        raise ValueError('rewards must be a nonempty finite sequence')
    mean = sum(rewards)/len(rewards)
    sd = math.sqrt(sum((x-mean)**2 for x in rewards)/len(rewards))
    return [(x-mean)/sd if sd else 0 for x in rewards]

def check_median(candidate):
    """Check trusted local callables on finite examples, not arbitrary code."""
    cases = [([1, 9], 5), ([1, 2, 9], 2), ([9, 1], 5),
             ([-4, -2], -3), ([2, 2, 2, 2], 2), ([7], 7)]
    failures = []
    for values, expected in cases:
        supplied = values.copy()
        try:
            actual = candidate(supplied)
            if actual != expected:
                failures.append({'input': values, 'kind': 'value',
                                 'expected': expected, 'actual': actual})
        except Exception as error:
            failures.append({'input': values, 'kind': 'exception',
                             'error': type(error).__name__})
        if supplied != values:
            failures.append({'input': values, 'kind': 'mutation'})
    return {'passed': not failures, 'failures': failures,
            'cases_checked': len(cases)}

def uct(mean, parent_visits, visits, exploration):
    if not all(type(v) is int for v in (parent_visits, visits)):
        raise ValueError('visit counts must be integers')
    if not 0 <= visits <= parent_visits:
        raise ValueError('require 0 <= visits <= parent visits')
    if not all(math.isfinite(v) for v in (mean, exploration)):
        raise ValueError('mean and exploration must be finite')
    if exploration < 0:
        raise ValueError('exploration must be nonnegative')
    if visits == 0:
        return math.inf
    return mean + exploration * math.sqrt(math.log(parent_visits) / visits)

def work_span(durations, dependencies):
    """Compute ideal work and critical path for a finite DAG."""
    if set(durations) != set(dependencies):
        raise ValueError('duration and dependency nodes must match')
    if any(not math.isfinite(v) or v <= 0 for v in durations.values()):
        raise ValueError('durations must be positive and finite')
    if any(p not in durations for ps in dependencies.values() for p in ps):
        raise ValueError('unknown dependency')
    try:
        order = list(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        raise ValueError('dependencies contain a cycle') from error
    finish = {}
    for node in order:
        start = max((finish[p] for p in dependencies[node]), default=0)
        finish[node] = start + durations[node]
    return sum(durations.values()), max(finish.values(), default=0)

def group_signatures(signatures):
    groups = {}
    for name, signature in signatures.items():
        groups.setdefault(signature, []).append(name)
    return list(groups.values())

def mixed_group_probability(p, group_size):
    coverage(p, group_size)
    if group_size < 2:
        raise ValueError('a comparison group needs at least two samples')
    return 1 - p ** group_size - (1 - p) ** group_size

def main():
    checks = {}
    assert allocate_samples([.2, .05], 4) == [4, 0]
    assert allocate_samples([], 0) == []
    assert uct(.6, 20, 2, .5) > uct(.8, 20, 10, .5)
    assert work_span({'a': 2, 'b': 3}, {'a': [], 'b': ['a']}) == (5, 5)
    assert mixed_group_probability(.5, 4) == .875
    assert sorted(map(len, group_signatures({'a': (1,), 'b': (1,),
                                           'c': (2,)}))) == [1, 2]
    assert not check_median(lambda xs: sorted(xs)[len(xs)//2])['passed']
    assert check_median(statistics.median)['passed']
    assert not check_median(lambda xs: 1 / 0)['passed']
    def mutating_median(values):
        values.sort()
        return statistics.median(values)
    assert any(f['kind'] == 'mutation'
               for f in check_median(mutating_median)['failures'])
    for operation in [lambda: group_advantages([]),
                      lambda: horizon(60, 0, .8),
                      lambda: allocate_samples([], 1),
                      lambda: mixed_group_probability(.5, 1),
                      lambda: uct(.5, 2, 3, .5),
                      lambda: work_span({'a': 1}, {'a': ['a']})]:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input was accepted')
    def check(name, actual, expected, tol=1e-4):
        assert math.isclose(actual, expected, abs_tol=tol), (name, actual, expected)
        checks[name] = actual
    check('100 samples at p=.02', coverage(.02,100), .8674)
    assert math.ceil(math.log(.05)/math.log(.98)) == 149
    assert math.ceil(math.log(.1)/math.log(.999)) == 2302
    assert math.ceil(math.log(.1)/math.log(.998)) == 1151
    check('pass@3: 2 correct among 10', pass_at_k(10,2,3), 8/15)
    check('uniform task mixture failure at k=10', 1/11, .090909)
    check('selected success: coverage times conditional selection', coverage(.1,20)*.7, .6149)
    check('rare-correct acceptance precision', acceptance_precision(.02,.9,.05), .2687)
    check('improved verifier precision', acceptance_precision(.02,.9,.001), .9484)
    check('false acceptance among 100 wrong candidates', coverage(.01,100), .6340)
    check('hard rollout label probability at p=.1 N=4', coverage(.1,4), .3439)
    check('32 rollout labels with success .1', coverage(.1,32), .9657)
    check('all 20 steps valid: independent .98', .98**20, .6676)
    check('mixed reward group: p=.01 G=8', 1-.01**8-.99**8, .0773)
    check('ten-stage success with recovery', .992**10, .9228)
    check('hundred-stage success with recovery', .9972**100, .7555, 2e-4)
    check('80 percent horizon', horizon(60,1,.8),15)
    check('95 percent horizon', horizon(60,1,.95),60/19)
    check('one-recovery ten-stage success', .972**10, .7528)
    check('one-recovery twenty-stage success', .9944**20, .8938)
    check('cross entropy truthful score', -.8*math.log(.8)-.2*math.log(.2), .5004)
    check('cross entropy overconfident score', -.8*math.log(.99)-.2*math.log(.01), .9291)
    check('expected distinct behaviors', 2-.1**10-.9**10, 1.6513)
    check('paired comparison standard error', math.sqrt((29/99)/100), .0541)
    check('effective sample size', 20/(1+19*.1), 6.89655)
    check('mixed group p=.1 G=4', mixed_group_probability(.1,4), .3438)
    check('post-training concentrated pass@20', (2-.1**20-.999**20)/2, .509906)
    check('pre-training diverse pass@20', (2-.5**20-.9**20)/2, .939211)
    assert group_advantages([1,0,1,0]) == [1,-1,1,-1]
    assert group_advantages([1,1,1,1]) == [0,0,0,0]
    assert coverage(0,10) == 0 and coverage(1,10) == 1 and coverage(.5,0) == 0
    assert pass_at_k(10,0,3) == 0 and pass_at_k(10,10,3) == 1
    assert acceptance_precision(.2,0,0) is None
    out = Path(__file__).resolve().parent/'validation'
    out.mkdir(exist_ok=True)
    (out/'numerical-checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(f'{len(checks)} numerical examples and boundary cases passed.')

if __name__ == '__main__':
    main()
