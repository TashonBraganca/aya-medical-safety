"""Start a function of the deployed app without creating an ephemeral app:  python launch.py <function> [key=value ...]"""
import sys, modal
kw = {}
for a in sys.argv[2:]:
    k, v = a.split("=", 1); kw[k] = {"true": True, "false": False}.get(v.lower(), int(v) if v.isdigit() else v)
c = modal.Function.from_name("lexsi-p9-causal", sys.argv[1]).spawn(**kw); print("spawned", sys.argv[1], kw, c.object_id)
