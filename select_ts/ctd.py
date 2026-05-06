from sympy.ntheory import primefactors


# Get unique prime factors as a list
# Output: [2, 5]


def find_min_ts(k: int):
    prime_factors = primefactors(k)
    ts = [(k, 1)]
    for i, p_curr in enumerate(prime_factors):
        curr_val = k // p_curr
        ts.append((curr_val, 0))
    return ts


if __name__ == "__main__":
    for c in [1, 5, 7, 10, 20, 27]:
        print(find_min_ts(c))
