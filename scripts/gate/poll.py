"""Status of OUR OWN submissions only (ids given on the command line)."""
import os, sys, json
from whestbench import aicrowd_client as ac
cls = [getattr(ac, n) for n in dir(ac) if n.endswith("Client") and isinstance(getattr(ac, n), type)][0]
c = cls(api_key=os.environ["AICROWD_API_KEY"]) if "api_key" in cls.__init__.__code__.co_varnames else cls(os.environ["AICROWD_API_KEY"])
for sid in sys.argv[1:]:
    r = c.get_submission_status(int(sid))
    keep = {k: r.get(k) for k in ("id", "grading_status_cd", "score", "score_secondary", "grading_message", "media_content_type") if k in r}
    meta = r.get("meta") or {}
    print(sid, json.dumps(keep)[:400], json.dumps(meta)[:600] if meta else "")
