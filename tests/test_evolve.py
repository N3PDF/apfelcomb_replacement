import copy

import numpy as np
import pineappl
import pytest
import yaml
from banana.data.theories import default_card
from ekobox.cards import example

import pineko.evolve


def test_sv_scheme():
    wrong_tcard = {"XIF": 1.0, "ModSV": "expanded"}
    schemeA_tcard = {
        "XIF": 2.0,
        "ModSV": "exponentiated",
    }
    schemeB_tcard = {"XIF": 0.5, "ModSV": "expanded"}
    schemeC_tcard = {"XIF": 2.0, "ModSV": None}
    with pytest.raises(ValueError):
        pineko.evolve.sv_scheme(wrong_tcard)
    assert pineko.evolve.sv_scheme(schemeA_tcard) == "exponentiated"
    assert pineko.evolve.sv_scheme(schemeB_tcard) == "expanded"
    assert pineko.evolve.sv_scheme(schemeC_tcard) is None


class FakeEvolInfo:
    x1 = [0.1, 1.0]
    fac1 = np.array([10.0])


class FakePine:
    _evolve_info = FakeEvolInfo()

    def orders(self):
        return [pineappl.boc.Order(0, 0, 0, 0, 0)]

    @property
    def convolutions(self):
        conv_type = pineappl.convolutions.ConvType(polarized=False, time_like=False)
        return [pineappl.convolutions.Conv(convolution_types=conv_type, pid=2212)]

    def channels(self):
        return [[([21, 21], 1)]]

    def evolve_info(self, _):
        return self._evolve_info

    @property
    def metadata(self):
        return {"convolution_particle_1": 2212, "convolution_particle_2": 11}


def test_write_operator_card_q0(tmp_path):
    """Checks https://github.com/NNPDF/pineko/issues/146"""
    p = tmp_path / "q0.yaml"
    g = FakePine()
    t = copy.deepcopy(default_card)
    o = copy.deepcopy(example.raw_operator())
    # 1. Same Q0 and mu0, all ok
    t["Q0"] = 5.0
    o["mu0"] = 5.0
    _xs, _mu2s = pineko.evolve.write_operator_card(g, o, p, t)
    with open(p, encoding="utf8") as f:
        oo = yaml.safe_load(f)
    np.testing.assert_allclose(oo["mu0"], t["Q0"])
    # 2. Q0 only in theory, all ok
    o.pop("mu0")
    _xs, _mu2s = pineko.evolve.write_operator_card(g, o, p, t)
    with open(p, encoding="utf8") as f:
        oo = yaml.safe_load(f)
    np.testing.assert_allclose(oo["mu0"], t["Q0"])
    # 3. op is different, raises error
    o["mu0"] = 11.0
    with pytest.raises(ValueError):
        _xs, _mu2s = pineko.evolve.write_operator_card(g, o, p, t)
