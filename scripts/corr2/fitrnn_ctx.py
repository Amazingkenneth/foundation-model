"""fitrnn.py <data tag> <train spec> <test spec> <epochs> <H> <S> <out>: the carried corrector with a learned S-dim
per-neuron state carried through W (18254-style). At layer l the MLP reads [X_l standardized, layer one-hot,
W_l^T state_{l-1} / sqrt(n * mean W^2)] and returns delta_l (fit to -C_l, layer weights = propagation gain or rho^(15-l))
and state_l. S = 0 is the plain per-layer corrector trained the same way (one network per step).
Weights are regenerated per network (hf_weights) in DataLoader workers."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr")
from prep import ids
tag, trs, tes = sys.argv[1:4]
EP, H, S = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]); OUT = sys.argv[7]
S2 = int(os.environ.get("S2", "0"))      # a second state channel carried through W*W (variance-like messages)
S3 = int(os.environ.get("S3", "0"))      # a third channel carried through W*W*W (kappa3-like messages)
NT = int(os.environ.get("NT", "2")); NW = int(os.environ.get("NW", "2")); torch.set_num_threads(NT)
torch.manual_seed(int(os.environ.get("SEED", "0")))
WMODE = os.environ.get("WMODE", "gain"); RHO = float(os.environ.get("RHO", "0.75")); ACT = os.environ.get("ACT", "gelu")
LR = float(os.environ.get("LR", "2e-3")); DEC = float(os.environ.get("DEC", "0.8"))
LOSS = os.environ.get("LOSS", "layer")
MIXW = float(os.environ.get("MIXW", "0.7"))   # mixp: weight of the transported (final-layer) term   # layer | e2e (final error after propagation) | mix (layer + e2e)
GAIN = [0.031, 0.052, 0.076, 0.107, 0.139, 0.180, 0.223, 0.282, 0.349, 0.412, 0.502, 0.580, 0.656, 0.765, 0.883, 1.0]
L = 16
D = f"/tmp/claude-0/sc/corr/data_{tag}"
tr = [k for k in ids(trs) if os.path.exists(f"{D}/{k}.npz")]; te = [k for k in ids(tes) if os.path.exists(f"{D}/{k}.npz")]
z0 = np.load(f"{D}/{tr[0]}.npz")
NAMES = [str(x) for x in z0["names"]] if "names" in z0.files else [str(i) for i in range(z0["X"].shape[2])]
DROP = [x for x in os.environ.get("DROP", "").split(",") if x]    # feature names (or column indices) to leave out
KEEP = np.array([i for i, nm in enumerate(NAMES) if nm not in DROP and str(i) not in DROP]); del z0
ROOT = os.environ.get("ROOT", "0") == "1"   # pk2v, pk3v, pk4v (raw columns 7, 8, 9) -> signed k-th roots (heavy tails)
ROOTC = [(int(np.where(KEEP == c)[0][0]), k) for c, k in ((7, 2), (8, 3), (9, 4)) if c in KEEP] if ROOT else []
def rootx(X):
    for c, k in ROOTC: X[..., c] = np.sign(X[..., c]) * np.abs(X[..., c]) ** (1.0 / k)
    return X
print(f"train {len(tr)} test {len(te)} H {H} S {S} S2 {S2} S3 {S3} {WMODE} {ACT} nx {len(KEEP)}/{len(NAMES)}"
      f"{' drop ' + ','.join(DROP) if DROP else ''}", flush=True)
# standardisation from (up to) 200 training networks
acc = []
for k in tr[:200]:
    z = np.load(f"{D}/{k}.npz"); acc.append((rootx(z["X"][..., KEEP].astype(np.float64)).reshape(-1, len(KEEP)), z["C"].astype(np.float64).ravel()))
PLSTD = os.environ.get("PLSTD", "0") == "1"   # per-layer input standardisation (mu, sd of shape (L, nx))
TANH = os.environ.get("TANH", "0") == "1"     # tanh on the carried states
TNORM = os.environ.get("TNORM", "0") == "1"   # delta_l = out * lsc_l (per-layer target rms; same objective)
CTX = int(os.environ.get("CTX", "0"))   # network context: 1 = per-layer mean over neurons of the inputs; 2 = + pooled carried states
IC = int(os.environ.get("IC", "0"))   # incoming-correction inputs: Phi_l o (W_l s_{l-1}) and W_l s_{l-1} (ysd units)
Xa = np.concatenate([a for a, _ in acc]); mu = Xa.mean(0).astype(np.float32); sd = (Xa.std(0) + 1e-12).astype(np.float32)
ysd = float(np.concatenate([b for _, b in acc]).std())
nx = mu.shape[0]
if PLSTD:
    X4 = Xa.reshape(len(acc), L, -1, nx)
    mu = X4.mean(axis=(0, 2)).astype(np.float32); sd = (X4.std(axis=(0, 2)) + 1e-12).astype(np.float32); del X4
C4 = np.concatenate([b for _, b in acc]).reshape(len(acc), L, -1)
lsc = (C4.std(axis=(0, 2)) / ysd).astype(np.float32) if TNORM else np.ones(L, np.float32); del C4
del acc, Xa
mu_b = mu[:, None, :] if mu.ndim == 2 else mu; sd_b = sd[:, None, :] if sd.ndim == 2 else sd
LAY0 = E2E0 = 1.0
if LOSS == "mixp":
    _l0, _e0 = [], []
    for k in tr[:200]:
        z = np.load(f"{D}/{k}.npz"); C_ = z["C"].astype(np.float64) / ysd; E_ = z["E"][-1].astype(np.float64) / ysd
        _l0.append(((C_ ** 2).mean(1) @ np.array(GAIN if WMODE == "gain" else [RHO ** (L - 1 - l) for l in range(L)])) / (sum(GAIN) if WMODE == "gain" else 1.0))
        _e0.append((E_ ** 2).mean())
    LAY0, E2E0 = float(np.mean(_l0)), float(np.mean(_e0)); print(f"mixp normalisers LAY0 {LAY0:.4g} E2E0 {E2E0:.4g} MIXW {MIXW}", flush=True)
print(f"PLSTD {PLSTD} TANH {TANH} TNORM {TNORM} lsc {np.array2string(lsc, precision=2, max_line_width=200)}", flush=True)
wl = torch.tensor(GAIN if WMODE == "gain" else [RHO ** (L - 1 - l) for l in range(L)], dtype=torch.float32)


class DS(torch.utils.data.Dataset):
    def __init__(self, keys): self.keys = keys
    def __len__(self): return len(self.keys)
    def __getitem__(self, i):
        from regen import hf_weights
        z = np.load(f"{D}/{self.keys[i]}.npz")
        X = ((rootx(z["X"][..., KEEP].astype(np.float32)) - mu_b) / sd_b).astype(np.float32)
        Ws = np.stack(hf_weights(int(z["seed"])))                              # (L, n, n) float32
        return (torch.from_numpy(X), torch.from_numpy((-z["C"] / ysd).astype(np.float32)), torch.from_numpy(Ws),
                torch.from_numpy(z["Phi"].astype(np.float32)), torch.from_numpy(z["E"].astype(np.float32)))


A = torch.nn.ReLU if ACT == "relu" else torch.nn.GELU


class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.f = torch.nn.Sequential(torch.nn.Linear(nx + L + S + S2 + S3 + 2 * IC + (nx if CTX else 0) + ((S + S2) if CTX >= 2 else 0), H), A(), torch.nn.Linear(H, H), A(),
                                  torch.nn.Linear(H, H), A(), torch.nn.Linear(H, 1 + S + S2 + S3))
    def forward(s, X, Ws, Phi=None):                                          # X (L, n, nx), Ws (L, n, n)
        n = X.shape[1]; eye = torch.eye(L); out = []; st = None; st2 = None; st3 = None; sc = None
        for l in range(L):
            parts = [X[l], eye[l].expand(n, L)]
            if CTX:
                parts.append(X[l].mean(0, keepdim=True).expand(n, X.shape[2]))
            if CTX >= 2:
                if S: parts.append((torch.zeros(1, S) if st is None else st.mean(0, keepdim=True)).expand(n, S))
                if S2: parts.append((torch.zeros(1, S2) if st2 is None else st2.mean(0, keepdim=True)).expand(n, S2))
            if IC:
                if sc is None: tw = torch.zeros(n); a_in = torch.zeros(n)
                else: tw = Ws[l].T @ sc; a_in = Phi[l] * tw
                parts += [a_in[:, None], tw[:, None]]
            if S:
                cin = torch.zeros(n, S) if st is None else (Ws[l].T @ st) * Ws_scale(Ws[l])
                parts.append(cin)
            if S2:
                cin2 = torch.zeros(n, S2) if st2 is None else ((Ws[l] * Ws[l]).T @ st2) / (W2sum(Ws[l]) / n)
                parts.append(cin2)
            if S3:
                W3 = Ws[l] * Ws[l] * Ws[l]
                cin3 = torch.zeros(n, S3) if st3 is None else (W3.T @ st3) / torch.sqrt((W3 * W3).sum() / n)
                parts.append(cin3)
            o = s.f(torch.cat(parts, 1)); out.append(o[:, 0] * float(lsc[l]))
            if IC: sc = out[-1] if sc is None else a_in + out[-1]       # the propagated correction so far
            sq = torch.tanh if TANH else (lambda t: t)
            if S: st = sq(o[:, 1:1 + S])
            if S2: st2 = sq(o[:, 1 + S:1 + S + S2])
            if S3: st3 = sq(o[:, 1 + S + S2:])
        return torch.stack(out)                                               # (L, n)


def W2sum(W):
    return (W * W).sum()


def Ws_scale(W):                                                              # W^T st has ~ n * mean(W^2) x |st|^2
    return 1.0 / torch.sqrt(W.shape[0] * (W * W).mean())


net = Net()
INIT = os.environ.get("INIT", "")   # fine-tune from a fitrnn model (its standardisation is kept)
if INIT:
    ck0 = torch.load(INIT, weights_only=False); net.load_state_dict(ck0["state"])
    mu, sd, ysd = ck0["mu"], ck0["sd"], ck0["ysd"]; assert ck0.get("names", None) in (None, [NAMES[i] for i in KEEP])
    print(f"init from {INIT}", flush=True)
opt = torch.optim.Adam(net.parameters(), lr=LR)
EMA = float(os.environ.get("EMA", "0"))   # exponential moving average of the weights (per step), used for the output
ema = {k: v.detach().clone() for k, v in net.state_dict().items()} if EMA else None; nstep = 0
CK = OUT + ".ckpt"; ep0 = 0          # per-epoch checkpoint: a container restart resumes at the next epoch
if os.path.exists(CK):
    ck = torch.load(CK, weights_only=False)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); ep0 = ck["ep"] + 1
    if EMA: ema = ck["ema"]; nstep = ck["nstep"]
    torch.set_rng_state(ck["rng"])
    print(f"resumed after epoch {ck['ep']}", flush=True)
dl = torch.utils.data.DataLoader(DS(tr), batch_size=None, shuffle=True, num_workers=NW, persistent_workers=True, prefetch_factor=2)
for ep in range(ep0, EP):
    tot, tb = 0.0, 0.0
    for X, Y, Ws, Phi, E in dl:
        p = net(X, Ws, Phi)
        lay = ((p - Y) ** 2).mean(1) @ wl / wl.sum(); lay0 = ((Y ** 2).mean(1) @ wl / wl.sum()).detach()
        if LOSS == "layer":
            loss = lay; base0 = lay0
        elif LOSS == "mixp":                                  # mliston (18170): (1-MIXW) per-layer + MIXW transported
            sL = torch.zeros(X.shape[1])
            for l in range(L):
                sL = (Phi[l] * (Ws[l].T @ sL) if l > 0 else 0.0) + p[l]
            eL = E[-1] / ysd
            e2 = ((eL + sL) ** 2).mean(); e20 = (eL ** 2).mean().detach()
            loss = (1 - MIXW) * lay / LAY0 + MIXW * e2 / E2E0; base0 = ((1 - MIXW) * lay0 / LAY0 + MIXW * e20 / E2E0).detach()
        elif LOSS == "e2ep":                                  # pooled final-layer MSE (the scored quantity)
            sL = torch.zeros(X.shape[1])
            for l in range(L):
                sL = (Phi[l] * (Ws[l].T @ sL) if l > 0 else 0.0) + p[l]
            eL = E[-1] / ysd
            loss = ((eL + sL) ** 2).mean(); base0 = (eL ** 2).mean().detach()
        else:
            sL = torch.zeros(X.shape[1])                      # propagated correction, in units of ysd
            for l in range(L):
                sL = (Phi[l] * (Ws[l].T @ sL) if l > 0 else 0.0) + p[l]
            eL = E[-1] / ysd
            e2e = ((eL + sL) ** 2).mean() / (eL ** 2).mean()
            loss = e2e if LOSS == "e2e" else lay / lay0 + e2e; base0 = torch.tensor(1.0 if LOSS == "e2e" else 2.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if EMA:
            nstep += 1; dcy = min(EMA, (1 + nstep) / (10 + nstep))
            with torch.no_grad():
                for k, v in net.state_dict().items(): ema[k].mul_(dcy).add_(v, alpha=1 - dcy)
        tot += loss.item(); tb += float(base0)
    for g in opt.param_groups: g["lr"] *= DEC
    print(f"epoch {ep}: weighted loss / no-fit {tot / tb:.4f}", flush=True)
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), ep=ep, rng=torch.get_rng_state(), ema=ema, nstep=nstep), CK + ".tmp")
    os.replace(CK + ".tmp", CK)
def test(lbl):
    base, corr = [], []
    with torch.no_grad():
        for X, Y, Ws, Phi, E in torch.utils.data.DataLoader(DS(te), batch_size=None, shuffle=False, num_workers=NW):
            dlt = net(X, Ws, Phi).double() * ysd; Phi = Phi.double(); W64 = Ws.double(); s = torch.zeros(X.shape[1], dtype=torch.float64)
            for l in range(L):
                s = (Phi[l] * (W64[l].T @ s) if l > 0 else 0.0) + dlt[l]
            eL = E[-1].double(); base.append(float((eL ** 2).mean())); corr.append(float(((eL + s) ** 2).mean()))
    base, corr = np.array(base), np.array(corr)
    ev = np.array([int(k[1:]) % 2 == 0 for k in te])
    print(f"TEST{lbl} final-layer dMSE {corr.sum()/base.sum()-1:+.2%} (even {corr[ev].sum()/base[ev].sum()-1:+.2%} "
          f"odd {corr[~ev].sum()/base[~ev].sum()-1:+.2%}), better {np.sum(corr < base)}/{len(te)}", flush=True)
def save(path):
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, ysd=ysd, H=H, S=S, S2=S2, S3=S3, nx=nx, act=ACT,
                    names=[NAMES[i] for i in KEEP], root=ROOTC, tanh=TANH, lsc=lsc, ic=IC, ctx=CTX), path)
if EMA:
    save(OUT + ".raw"); test(" raw"); net.load_state_dict(ema)
save(OUT); test(" ema" if EMA else "")
