import random
import time
from sympy import isprime, nextprime, legendre_symbol
from statistics import mean


# Tonelli-Shanks algorithm for modular square root
def modular_sqrt(a, p):
    if legendre_symbol(a, p) != 1:
        return None
    elif a == 0:
        return 0
    elif p % 4 == 3:
        return pow(a, (p + 1) // 4, p)

    s, q = 0, p - 1
    while q % 2 == 0:
        q //= 2
        s += 1

    for z in range(2, p):
        if legendre_symbol(z, p) == -1:
            break

    c = pow(z, q, p)
    x = pow(a, (q + 1) // 2, p)
    t = pow(a, q, p)
    m = s

    while t != 1:
        for i in range(1, m):
            if pow(t, 2 ** i, p) == 1:
                break
        b = pow(c, 2 ** (m - i - 1), p)
        x = x * b % p
        t = t * b * b % p
        c = b * b % p
        m = i

    return x


def generate_curve_params(bits):
    while True:
        P = nextprime(random.getrandbits(bits))
        a = random.randrange(P)
        b = random.randrange(P)
        if (4 * pow(a, 3, P) + 27 * pow(b, 2, P)) % P != 0:
            return P, a, b


def generate_point_on_curve_fast(a, b, p):
    while True:
        x = random.randrange(p)
        rhs = (x ** 3 + a * x + b) % p
        y = modular_sqrt(rhs, p)
        if y is not None:
            return x, y


def point_add(P, Q, a, p):
    if P is None:
        return Q
    if Q is None:
        return P
    if P == Q:
        if P[1] == 0:
            return None
        s = (3 * P[0] ** 2 + a) * pow(2 * P[1], -1, p)
    else:
        if P[0] == Q[0]:
            return None
        s = (Q[1] - P[1]) * pow(Q[0] - P[0], -1, p)

    s %= p
    x_r = (s ** 2 - P[0] - Q[0]) % p
    y_r = (s * (P[0] - x_r) - P[1]) % p
    return x_r, y_r


def scalar_mult(k, P, a, p):
    R = None
    while k:
        if k & 1:
            R = point_add(R, P, a, p)
        P = point_add(P, P, a, p)
        k >>= 1
    return R


def simulate_key_exchange_optimized(bits, trials=5):
    times_A, times_B, times_R = [], [], []

    for _ in range(trials):
        P, a, b = generate_curve_params(bits)
        G = generate_point_on_curve_fast(a, b, P)

        Ka = random.randrange(1, P)
        Kb = random.randrange(1, P)

        t0 = time.time()
        A = scalar_mult(Ka, G, a, P)
        t1 = time.time()
        B = scalar_mult(Kb, G, a, P)
        t2 = time.time()
        R = scalar_mult(Ka, B, a, P)
        t3 = time.time()

        times_A.append(t1 - t0)
        times_B.append(t2 - t1)
        times_R.append(t3 - t2)

    return {
        'k': bits,
        'A': mean(times_A),
        'B': mean(times_B),
        'shared key R': mean(times_R)
    }


def main():
    for bits in [128, 192, 256]:
        result = simulate_key_exchange_optimized(bits)
        print(f"Key Size: {result['k']} bits")
        print(f"  Time for A: {result['A']:.6f} s")
        print(f"  Time for B: {result['B']:.6f} s")
        print(f"  Time for Shared Key R: {result['shared key R']:.6f} s\n")


if __name__ == "__main__":
    main()
