from dataclasses import replace

import pytest

from gls1dr.cli import _default_signatures
from gls1dr.config import Config
from gls1dr.signatures import load_signatures
from gls1dr.synthetic import make_example


@pytest.fixture(scope="session")
def spec():
    return load_signatures(_default_signatures())


@pytest.fixture(scope="session")
def cfg():
    # small sizes: the tests check logic, not statistical precision
    return replace(Config(), n_boot=60, n_mc=2000, n_verify=1000, n_ode_checks=20)


@pytest.fixture(scope="session")
def ds(spec, cfg):
    return make_example(spec, cfg, seed=1)
