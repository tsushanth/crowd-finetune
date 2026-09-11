import hashlib
import hmac
import json
import time
import urllib.parse


def _parse_pairs(init_data):
    pairs = {}
    raw = {}
    for chunk in init_data.split("&"):
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        raw[key] = value
        pairs[key] = urllib.parse.unquote(value)
    return pairs, raw


def verify(init_data, bot_token, max_age_seconds=86400):
    if not init_data or not bot_token:
        return None
    try:
        pairs, raw = _parse_pairs(init_data)
        if "hash" not in pairs or "auth_date" not in pairs or "user" not in pairs:
            return None
        auth_date = int(pairs["auth_date"])
        if time.time() - auth_date > max_age_seconds:
            return None
        data_check_string = "\n".join(
            f"{k}={v}" for k, v in sorted(pairs.items()) if k != "hash"
        )
        secret_key = hmac.new(
            key=b"WebAppData", msg=bot_token.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        expected_hash = hmac.new(
            key=secret_key, msg=data_check_string.encode("utf-8"), digestmod=hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_hash, raw.get("hash", "")):
            return None
        return {
            "hash": raw.get("hash", ""),
            "auth_date": pairs["auth_date"],
            "user": json.loads(pairs["user"]),
            "raw": init_data,
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def user_id_from_message(verified):
    return str(verified["user"]["id"])


def username_from_message(verified):
    user = verified["user"]
    return user.get("username") or user.get("first_name", "")