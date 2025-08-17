import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_PINECONE_TESTS"),
    reason="Pinecone tests are optional; set RUN_PINECONE_TESTS=1 to enable.",
)


def test_placeholder():
    # Real Pinecone integration tests live in standalone scripts or manual runs.
    # This placeholder keeps CI green while documenting how to enable them.
    assert True
