from __future__ import division
from random import random
import matplotlib.pyplot as plt
import numpy as np


def integrate(f, lo, hi, steps=200):
    dx = (hi - lo) / steps
    lo += dx / 2
    return sum(f(i * dx + lo) * dx for i in range(steps))


def make_cdf(f, lo, hi, steps=200):
    total_area = integrate(f, lo, hi, steps)

    def cdf(x):
        assert lo <= x <= hi
        return integrate(f, lo, x, steps) / total_area
    return cdf


def bisect(target, f, lo, hi, n=30):
    'Find x between lo and hi where f(x)=target'
    old_value = -99999
    abs_error = 1E-2
    for i in range(n):
        mid = (hi + lo) / 2.0
        if target < f(mid):
            hi = mid
        else:
            lo = mid
        new_value = (hi + lo) / 2.0
        if abs(old_value - new_value) <= abs_error:
            return new_value
        old_value = new_value
    return new_value


def make_user_distribution(f, lo, hi, steps=50, n=15):
    def linear(x):
        return np.exp(f.score_samples(np.array(x).reshape(1, -1)))

    cdf = make_cdf(linear, lo, hi, steps)
    result = []
    for i in range(n):
        sample = bisect(random(), cdf, lo, hi, n)
        result.append(sample)
        print("created sample " + str(i) + ": "+str(sample))
    return result

    # return lambda: bisect(random(), cdf, lo, hi, n)


def percentile(p, f, lo, hi, steps=200, n_bisect=100):
    def linear(x):
        return np.exp(f.score_samples(np.array(x).reshape(1, -1)))

    cdf = make_cdf(linear, lo, hi, steps)

    return bisect(p, cdf, lo, hi, n_bisect)



if __name__ == '__main__':

    def linear(x):
        return 3 * x - 6
    lo, hi = 2, 10
    r = make_user_distribution(linear, lo, hi)

    plt.plot(r)
    plt.show()
