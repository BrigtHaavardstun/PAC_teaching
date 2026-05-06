
def dp(c, S) -> list[float]:
    pass


def main(p, q, G, B, C: list[int], X, S, df):
    k = len(S)
    PMFc = [[0.0]]*len(C)
    P_eq_c = [[0.0]*k for _ in C]
    # Step 1, calc probability for concept being selected 0..k times.
    for c in C:
        res = dp(c, S)
        PMFc[c] = res
        for i in range(k+1):
            P_eq_c[c][i] = res[i]

    # Step 2, calc probability for concept being less than or equal 0..k
    P_leq_c = [[0.0]*k] * len(C)
    for c in C:
        P_leq_c[c][0] = P_eq_c[c][0]
        for i in range(1, k+1):
            P_leq_c[c][i] = P_eq_c[c][i] + P_leq_c[c][i-1]

    # Step 3, calc probabilty for max concept in G and B.
