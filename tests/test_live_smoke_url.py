"""live_smoke.py URL composition + subprotocol auth regressions (offline).

live_smoke previously appended `model=&token=<token>` unconditionally: any
user-provided ?model= was clobbered (real set bargein/cancel checks were
impossible) and the instance token leaked into the URL/access logs.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from live_smoke import compose_ws_url, ws_subprotocols  # noqa: E402


def test_default_model_empty_unchanged_behavior():
    assert compose_ws_url("ws://host:8765/v1/realtime") == "ws://host:8765/v1/realtime?model="


def test_model_flag_wins_over_url():
    url = compose_ws_url("ws://host:8765/v1/realtime?model=profile:old", "profile:new")
    assert url == "ws://host:8765/v1/realtime?model=profile%3Anew"


def test_existing_url_model_preserved_when_flag_omitted():
    url = compose_ws_url("ws://host:8765/v1/realtime?model=profile:anna")
    assert url == "ws://host:8765/v1/realtime?model=profile%3Aanna"


def test_other_query_params_survive():
    url = compose_ws_url("ws://host:8765/v1/realtime?foo=bar&model=x", "profile:y")
    assert url == "ws://host:8765/v1/realtime?foo=bar&model=profile%3Ay"
    url2 = compose_ws_url("ws://host:8765/v1/realtime?foo=bar")
    assert url2 == "ws://host:8765/v1/realtime?foo=bar&model="


def test_model_quoting_safe():
    url = compose_ws_url("ws://host/v1/realtime", "profile:a b&c=d")
    assert "profile%3Aa+b%26c%3Dd" in url  # fully query-encoded, no raw separators


def test_token_never_in_url_only_subprotocol():
    token = "secret-instance-token"
    subs = ws_subprotocols(token)
    assert subs == ["rtv.bearer", "rtv.bearer.secret-instance-token"]
    for url in (
        compose_ws_url("ws://host/v1/realtime"),
        compose_ws_url("ws://host/v1/realtime?model=profile:x", "profile:y"),
    ):
        assert token not in url
        assert "token=" not in url
