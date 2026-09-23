"""Post a signed test event to the Hermes webhook route from inside the bridge.

Run: docker exec <bridge-container> python3 /app/test_forward.py

Uses the same URL and secret the bridge itself holds, so a 200 here proves the
signature scheme, the URL and the container-to-container route all work before any
real SMS is involved.
"""
import hashlib
import hmac
import json
import os
import time
import urllib.request

url = os.environ.get("HERMES_WEBHOOK_URL", "")
secret = os.environ.get("HERMES_WEBHOOK_SECRET", "")
if not url or not secret:
    raise SystemExit("HERMES_WEBHOOK_URL / HERMES_WEBHOOK_SECRET not set")

mid = "selftest-%d" % int(time.time())
body = json.dumps({
    "event_type": "sms.received",
    "from": "+15144416963",
    "text": "Self-test from the bridge container. If this reaches the agent and Telegram, the inbound SMS path works.",
    "message_id": mid,
    "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, separators=(",", ":"))

ts = str(int(time.time()))
sig = hmac.new(secret.encode(), f"{ts}.{body}".encode(), hashlib.sha256).hexdigest()
req = urllib.request.Request(url, data=body.encode(), method="POST", headers={
    "Content-Type": "application/json",
    "X-Webhook-Signature-V2": sig,
    "X-Webhook-Timestamp": ts,
    "X-Request-ID": mid,
})
try:
    with urllib.request.urlopen(req, timeout=45) as r:
        print("POST", url)
        print("->", r.status, r.read().decode()[:300])
except Exception as exc:
    print("POST", url)
    print("-> FAILED:", type(exc).__name__, exc)
    raise SystemExit(1)
