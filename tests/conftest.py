import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LANSHARD_THREADS", "1")  # tiny-model workers: one thread each is fastest


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    from tiny_model import make_tiny_model

    return make_tiny_model(str(tmp_path_factory.mktemp("tiny-qwen2")))
