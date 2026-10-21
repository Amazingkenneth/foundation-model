"""patch_c79.py <estimator.py> <ROWF>: cn78 -> cn79, in place. Dropped final rows get scale-mode surrogates instead of zeros:
D3 = c mu sigma^2 (c fitted on the kept rows) and star = mean(star / sigma^4 over kept rows) sigma^4."""
import sys

p, rowf = sys.argv[1], sys.argv[2]
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:100]
    s = s.replace(old, new)


rep('''ROWF = float(_os.environ.get("ROWF", "0.27"))   # cn78: 768 of 1024 final rows keep D3 (mini m0-99: see ALGORITHM.md section 6)''',
    '''ROWF = float(_os.environ.get("ROWF", "%s"))   # cn79 (cn78: 0.27); see ALGORITHM.md section 6
# cn79: the dropped rows are not zeroed. The final standardized skew is almost a function of alpha (D3 / sigma^3 vs alpha:
# corr 0.95, the coherent norm mode of a bias-free net, D3 ~ c mu sigma^2), and the k34 star feature is almost a constant
# times sigma^4 (-0.014 +- 0.003): both are fitted on the kept rows and filled in (zero-cost; the k34 inputs of the dropped
# rows stay on their manifold).
ROWFILL = int(_os.environ.get("ROWFILL", "2"))''' % rowf)
rep('''                if rsel:
                    D3 = fnp.take(fnp.concatenate([D3, fnp.zeros(n - mr, dtype=f32)]), rs_inv)''',
    '''                if rsel and ROWFILL:
                    _vk = fnp.take(var, rs_idx)
                    _xk = fnp.take(mu, rs_idx) * _vk
                    _c = fnp.sum(D3 * _xk) / fnp.maximum(fnp.sum(_xk * _xk), 1e-30)
                    _drop = fnp.take(fnp.concatenate([fnp.zeros(mr, dtype=f32), fnp.ones(n - mr, dtype=f32)]), rs_inv)
                    D3 = fnp.take(fnp.concatenate([D3, fnp.zeros(n - mr, dtype=f32)]), rs_inv) + _drop * (mu * var) * _c
                elif rsel:
                    D3 = fnp.take(fnp.concatenate([D3, fnp.zeros(n - mr, dtype=f32)]), rs_inv)''')
rep('''                        if rsel and star is not None:
                            star = fnp.take(fnp.concatenate([star, fnp.zeros(n - mr, dtype=f32)]), rs_inv)''',
    '''                        if rsel and star is not None and ROWFILL >= 2:
                            _vk2 = fnp.take(var, rs_idx)
                            _sr = fnp.sum(star / (_vk2 * _vk2)) * (1.0 / mr)
                            star = (fnp.take(fnp.concatenate([star, fnp.zeros(n - mr, dtype=f32)]), rs_inv)
                                    + _drop * (var * var) * _sr)
                        elif rsel and star is not None:
                            star = fnp.take(fnp.concatenate([star, fnp.zeros(n - mr, dtype=f32)]), rs_inv)''')
open(p, "w").write(s)
print("patched")
