import os, httpx, logging
from typing import Tuple, Dict, Any

CF_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
CF_SECRET     = os.getenv("TURNSTILE_SECRET", "")
BYPASS        = os.getenv("TURNSTILE_DEV_BYPASS", "0") == "1"  # pour dev/local
TIMEOUT       = httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=10.0)

logger = logging.getLogger("turnstile")

async def verify_turnstile(token: str | None, remoteip: str | None = None) -> Tuple[bool, Dict[str, Any]]:
    # Mode dev/bypass (ex: pas d'Internet en local)
    if BYPASS:
        logger.warning("[turnstile] BYPASS=ON -> validation forcée côté serveur (dev only)")
        return True, {"bypass": True}

    # Clés de test Cloudflare (commencent par 1x...) => ça nécessite quand même Internet
    if (os.getenv("TURNSTILE_SITEKEY","").startswith("1x")
        and os.getenv("TURNSTILE_SECRET","").startswith("1x")):
        logger.info("[turnstile] Test keys in use")

    if not token:
        return False, {"reason": "missing-token"}
    if not CF_SECRET:
        logger.error("[turnstile] TURNSTILE_SECRET manquante")
        return False, {"reason": "missing-secret"}

    data = {"secret": CF_SECRET, "response": token}
    if remoteip:
        data["remoteip"] = remoteip

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(CF_VERIFY_URL, data=data)
        payload = r.json()
        ok = bool(payload.get("success"))
        if not ok:
            logger.warning("[turnstile] verify failed: %s", payload)
        return ok, payload
    except httpx.HTTPError as e:
        # ConnectTimeout, ReadTimeout, etc.
        logger.error("[turnstile] HTTP error: %r", e)
        return False, {"reason": "http-error", "error": str(e)}