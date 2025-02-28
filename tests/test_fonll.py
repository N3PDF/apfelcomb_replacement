import copy
import json
import pathlib

from banana.data.theories import default_card

import pineko


def test_FONLLInfo():
    full_list = [
        "ffns3.pineappl.lz4",
        "ffn03.pineappl.lz4",
        "ffns4til.pineappl.lz4",
        "ffns4bar.pineappl.lz4",
        "ffn04.pineappl.lz4",
        "ffns5til.pineappl.lz4",
        "ffns5bar.pineappl.lz4",
    ]
    fullfonll_fake_info = pineko.fonll.FONLLInfo(*full_list)
    wrongfonll_fake_info = pineko.fonll.FONLLInfo(
        full_list[0],
        full_list[1],
        None,
        full_list[3],
        full_list[4],
        None,
        None,
    )
    partialfonll_fake_info = pineko.fonll.FONLLInfo(
        full_list[0], full_list[1], full_list[2], full_list[3], None, None, None
    )
    name_list = [
        "ffns3",
        "ffn03",
        "ffns4zeromass",
        "ffns4massive",
        "ffn04",
        "ffns5zeromass",
        "ffns5massive",
    ]
    assert fullfonll_fake_info.fk_paths == {
        name: pathlib.Path(fk) for name, fk in zip(name_list, full_list)
    }
    assert wrongfonll_fake_info.fk_paths == {
        name: pathlib.Path(fk)
        for name, fk in zip(
            name_list[:2] + name_list[3:5],
            full_list[:2] + full_list[3:5],
        )
    }
    assert partialfonll_fake_info.fk_paths == {
        name: pathlib.Path(fk)
        for name, fk in zip(name_list[:4], full_list[:4])
        if fk is not None
    }


class FakeGrid:
    kv: dict = {}

    @property
    def metadata(self):
        return self.kv

    def set_metadata(self, k, v):
        self.kv[k] = v


def test_update_fk_theorycard(tmp_path):
    # prepare base card
    p = tmp_path / "blub.yaml"
    base_tc = copy.deepcopy(default_card)
    base_tc["PTO"] = 2
    p.write_text(json.dumps(base_tc))
    # fake grid
    fg = FakeGrid()
    fk_tc = copy.deepcopy(default_card)
    fk_tc["PTO"] = 1
    fg.set_metadata("theory", json.dumps(fk_tc))
    # run the update
    pineko.fonll.update_fk_theorycard(fg, p)
    # check it actually worked
    new_tc = json.loads(fg.metadata["theory"])
    assert new_tc["PTO"] == base_tc["PTO"]
