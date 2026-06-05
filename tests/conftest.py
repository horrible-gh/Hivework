"""Session-wide pytest setup.

Loads Hivework's out-of-repo secrets file (``~/.hivework/.env``, written by
``hive_setup``) into ``os.environ`` BEFORE any test runs — the same loader the CLI
uses (``hive.secrets.load_secrets``). So a token configured once in setup is referenced
directly by the test suite; live tests no longer depend on the launcher's
``.ai_launcher_secrets.bat`` having ``set`` the variable first.

Precedence is ``setdefault`` (real env wins), so this never clobbers a value already
exported by CI or a launcher, and it is harmless for the mocked unit tests that
``monkeypatch`` their own env.
"""
from hive.secrets import load_secrets

load_secrets()
