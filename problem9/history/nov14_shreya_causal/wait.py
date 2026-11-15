"""Block until a spawned call finishes:  python wait.py <call id> [timeout seconds]"""
import sys, modal
c = modal.FunctionCall.from_id(sys.argv[1])
try: print("RESULT", c.get(timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 540))
except TimeoutError: print("STILL_RUNNING")
except Exception as e: print("FAILED", type(e).__name__, str(e)[:1500])
