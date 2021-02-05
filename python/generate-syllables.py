
import itertools

C = """
c, d, f, h, k, l, m, n, p, q, r, s, t, v, x, y, z,
ch, dh, hl, hm, hn, kh, ml, nc, nk, np, nq, nr, nt, ny, ph, pl, qh, rh, th, ts, tx, xy,
hml, hny, nch, nkh, nph, npl, nqh, nrh, nth, nts, ntx, plh, tsh, txh,
nplh, ntsh, ntxh
""".strip()

V = """
a, e, i, o, u, w,
ai, au, aw, ee, ia, oo, ua
""".strip()

T = """
j, s, v, m, g, b
""".strip()

C = [c.strip() for c in C.split(',')]
V = [v.strip() for v in V.split(',')]
T = [t.strip() for t in T.split(',')]

print(f'C = {len(C)}, V = {len(V)}, T = {len(T)}')

c_frag = '|'.join([f"'{c}'" for c in C])
v_frag = '|'.join([f"'{c}'" for c in V])
t_frag = '|'.join([f"'{c}'" for c in T])

c_frag = f'({c_frag})'
v_frag = f'({v_frag})'
t_frag = f'({t_frag})'

print(c_frag)
print(v_frag)
print(t_frag)
