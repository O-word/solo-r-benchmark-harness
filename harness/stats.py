"""Statistics used in reports (stdlib only)."""
import math


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (rate, lo, hi); (nan,nan,nan) if n == 0."""
    if n <= 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, min(p, max(0.0, (c - h) / d)), max(p, min(1.0, (c + h) / d))


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cohen_kappa(a, b):
    """Unweighted Cohen's kappa for two equal-length label lists. Returns (kappa, observed_agreement, n)."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan"), float("nan"), n
    labels = sorted(set(a) | set(b), key=str)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if abs(1 - pe) < 1e-12:
        return (1.0 if po == 1 else float("nan")), po, n
    return (po - pe) / (1 - pe), po, n


def weighted_kappa(a, b, categories):
    """Quadratic-weighted kappa for ordinal labels (e.g. 1-5)."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) / (k - 1)) ** 2 if k > 1 else 0
            num += w * obs[i][j]
            den += w * ra[i] * cb[j] / n
    if den == 0:
        return float("nan")
    return 1 - num / den


def pearson(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)
