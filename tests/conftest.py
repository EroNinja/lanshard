import os
import sys

_status = {"code": 0}


def pytest_sessionfinish(session, exitstatus):
    _status["code"] = int(exitstatus)


def pytest_unconfigure(config):
    """gRPC servers and Gloo process groups leave non-daemon native threads behind, which keep the
    interpreter alive after the tests finish on Linux. On CI, exit hard once pytest has printed its
    summary (this hook runs last), preserving the exit code."""
    if os.environ.get("CI"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_status["code"])
