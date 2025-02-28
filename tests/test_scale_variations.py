import math

import numpy as np
from eko.beta import beta_qcd

from pineko import scale_variations


def test_ren_sv_coeffs():
    np.testing.assert_allclose(
        scale_variations.ren_sv_coeffs(m=0, max_as=0, logpart=0, which_part=0, nf=5), 0
    )
    np.testing.assert_allclose(
        scale_variations.ren_sv_coeffs(m=0, max_as=1, logpart=1, which_part=0, nf=5), 0
    )
    res_nf5 = scale_variations.ren_sv_coeffs(
        m=1, max_as=1, logpart=1, which_part=0, nf=5
    )
    res_nf4 = scale_variations.ren_sv_coeffs(
        m=1, max_as=1, logpart=1, which_part=0, nf=4
    )
    np.testing.assert_allclose(
        res_nf5 / res_nf4, beta_qcd((2, 0), 5) / beta_qcd((2, 0), 4)
    )
    exp_res = beta_qcd((2, 0), 5) * (1.0 / (4.0 * np.pi))
    np.testing.assert_allclose(
        scale_variations.ren_sv_coeffs(m=0, max_as=2, logpart=1, which_part=1, nf=5),
        exp_res,
    )
    np.testing.assert_allclose(
        scale_variations.ren_sv_coeffs(m=0, max_as=2, logpart=2, which_part=0, nf=5),
        0.0,
    )
    # check all coeff are set
    for m in (0, 1, 10):
        for max_as in range(1, 3 + 1):
            for logpart in range(1, max_as + 1):
                for which_part in range(0, max_as - logpart + 1):
                    for nf in (4, 5):
                        c = scale_variations.ren_sv_coeffs(
                            m=m,
                            max_as=max_as,
                            logpart=logpart,
                            which_part=which_part,
                            nf=nf,
                        )
                        assert np.isfinite(c)
                        if which_part > 0 or m > 0:
                            assert c > 0.0
                        else:
                            assert c >= 0.0
    # due to the exponential structure we can predict some things:
    for nf in (4, 5):
        # the highest log is always proportional beta0^k
        for k in range(1, 3 + 1):
            c = scale_variations.ren_sv_coeffs(
                m=1, max_as=k, logpart=k, which_part=0, nf=nf
            )
            bare_c = c * (4.0 * np.pi / beta_qcd((2, 0), nf)) ** k
            int_c = bare_c * math.factorial(k)
            np.testing.assert_allclose(int_c, int(int_c))
        # and even the second highest for the highest coeff
        for k in range(2, 3 + 1):
            c = scale_variations.ren_sv_coeffs(
                m=1, max_as=k, logpart=k - 1, which_part=1, nf=nf
            )
            bare_c = c * (4.0 * np.pi / beta_qcd((2, 0), nf)) ** (k - 1)
            int_c = bare_c * math.factorial(k)
            np.testing.assert_allclose(int_c, int(int_c))


def test_requirements():
    m = 0
    max_as = 1
    # In this case we expect only one necessary order
    exp_to_compute_ord = (1, 0, 1, 0, 0)
    exp_nec_order = (0, 0, 0, 0, 0)
    assert scale_variations.requirements(m, max_as, 0)[exp_to_compute_ord] == [
        exp_nec_order
    ]
