
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
j, s, v, m, g, b, _
""".strip()

C = [c.strip() for c in C.split(',')]
V = [v.strip() for v in V.split(',')]
T = [t.strip() for t in T.split(',')]

print(f'C = {len(C)}, V = {len(V)}, T = {len(T)}')

CV = [f'{c}{v}' for c, v in itertools.product(C, V)]
for cv in CV:
    cvt = [f'{cv}{t}'.replace('_', '') for cv, t in itertools.product([cv], T)]
    cvt = [f'`{i} <_static/mp3/{i}.mp3>`_' for i in cvt]
    cvt = ', '.join(cvt)
    print(cvt)

for v in V:
    vt = [f'{v}{t}'.replace('_', '') for v, t in itertools.product([v], T)]
    vt = [f'`{i} <_static/mp3/{i}.mp3>`_' for i in vt]
    vt =', '.join(vt)
    print(vt)