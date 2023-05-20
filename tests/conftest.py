import pytest

from audiozen.utils import set_random_seed


@pytest.fixture
def random():
    set_random_seed(3407)
