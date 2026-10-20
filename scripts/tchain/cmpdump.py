import sys, pickle, numpy as np
A = pickle.load(open(sys.argv[1], "rb")); B = pickle.load(open(sys.argv[2], "rb"))
def od(x): x = np.array(x, dtype=np.float64); np.fill_diagonal(x, 0); return x
def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
for l in range(min(len(A["dumps"]), len(B["dumps"]))):
    a, b = A["dumps"][l], B["dumps"][l]
    out = []
    out.append(("mu", rel(a["mu"], b["mu"])))
    out.append(("Cdiag", rel(np.diag(a["C"]), np.diag(b["C"]))))
    out.append(("Coff", rel(od(a["C"]), od(b["C"]))))
    out.append(("K4v", rel(a["K4v"], b["K4v"])))
    out.append(("K22", rel(od(a["K22"]), od(b["K22"]))))
    out.append(("K31", rel(od(a["K31"]), od(b["K31"]))))
    if a.get("K211c") is not None and b.get("K211_cap") is not None:
        out.append(("K211cap", rel(a["K211c"], od(b["K211_cap"]))))
        out.append(("pk211", rel(a["pk211"], od(b["pk211_cap"]))))
    out.append(("Gdiag", rel(np.diag(a["G"]), np.diag(b["G"]))))
    out.append(("Goff", rel(od(a["G"]), od(b["G"]))))
    print("layer", l, " ".join("%s %.2e" % kv for kv in out))
    print("   norms: Goff ours %.3e lean %.3e | K211cap ours %.3e lean %.3e | K31 ours %.3e lean %.3e" % (
        np.linalg.norm(od(a["G"])), np.linalg.norm(od(b["G"])),
        np.linalg.norm(a["K211c"]) if a.get("K211c") is not None else 0, np.linalg.norm(od(b["K211_cap"])),
        np.linalg.norm(od(a["K31"])), np.linalg.norm(od(b["K31"]))))
    if a.get("birth") is not None:
        print("   ours birth norm %.3e, cA*K211cap norm %.3e" % (np.linalg.norm(a["birth"]), np.linalg.norm(a["K211c"]) * 6 / 1028))
