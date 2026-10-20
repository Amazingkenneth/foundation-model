# Spherical-cubature test: final-layer MSE of point-set estimators vs 1e9-sample labels.
import sys, math, time, numpy as np, torch
torch.set_num_threads(2)
z = np.load("/tmp/claude-0/hfwork/mini.npz")
n, L = 1024, 16
Er = math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))   # E|x| for chi_n

def fwd(X, W):
    for l in range(L):
        X = torch.relu(X @ W[l])
    return X.mean(0)

def frames(k, g):
    out = []
    for _ in range(k):
        A = torch.randn(n, n, generator=g)
        Q, R = torch.linalg.qr(A)
        Q = Q * torch.sign(torch.diagonal(R))[None, :]
        out.append(Q.T)          # rows = orthonormal points
    P = torch.cat(out, 0) * Er
    return torch.cat([P, -P], 0)  # antipodal

def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)

H = hadamard(n)
def hframes(k, g):
    # H D_j: orthonormal frames from random sign diagonals (cheap structured rotations)
    out = []
    for _ in range(k):
        d = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
        out.append(H * d[None, :])   # rows orthonormal
    P = torch.cat(out, 0) * Er
    return torch.cat([P, -P], 0)

ids = [int(a) for a in sys.argv[1].split(",")]
Ns = [int(a) for a in sys.argv[2].split(",")]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
for i in ids:
    W = torch.from_numpy(np.load(f"/tmp/claude-0/cub/w_m{i}.npy"))
    j = int(np.where(z["mlp_id"] == i)[0][0])
    t = torch.from_numpy(z["means"][j][-1].astype(np.float32)).double(); nz = z["avg_var"][j] / 1e9
    for N in Ns:
        k = max(1, N // (2 * n))
        for kind in ("iid", "sph", "frames", "hframes"):
            res = []
            for rep in range(reps):
                g = torch.Generator().manual_seed(1000 * rep + 7)
                if kind == "iid":
                    X = torch.randn(N, n, generator=g)
                elif kind == "sph":
                    X = torch.randn(N, n, generator=g); X = X / X.norm(dim=1, keepdim=True) * Er
                elif kind == "frames":
                    X = frames(k, g)
                else:
                    X = hframes(k, g)
                t0 = time.time()
                p = fwd(X, W).double()
                res.append(float(((p - t) ** 2).mean()) - nz)
            print(f"m{i} N={N:6d} {kind:8s} mse {np.mean(res):.3e}  ({' '.join('%.2e' % r for r in res)})  avg_var {z['avg_var'][j]:.4f}", flush=True)
