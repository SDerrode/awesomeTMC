"""Exactness domains of the classical collapsed smoothers (experiment S13).

Short records (N = 6) against the exact K^N path smoother:

* the GPB2 pair-RTS of the generic collapsed smoother is the S7 pair-RTS;
* Kim's smoother and the IMM-RTS are exact on {C==0, A==0}; on {C==0} with
  state memory their regimes stay exact but their state is biased; on AB
  their regimes are biased, while the GPB2 pair-RTS is exact;
* the K^2 embedding of a slaved AB model transports the exact filter and
  the smoothed regimes, the K^2-mode IMM-RTS realizes the smoother of the
  embedded model, and that smoother is not the CGOMSM2 one;
* the one-step collapsed smoother is exact on the domain of its forward
  filter: the cross-annihilation family with the GPB2 bank, {C==0} with the
  IMM bank.
"""

import numpy as np
import pytest

from prg.experiments.cns_exactness import (
    SEED0,
    _rel,
    ab_model,
    cgo_memory_model,
    cross_annihilation_model,
)
from prg.experiments.reference_filters import exact_mixture_filter
from prg.experiments.smoothing_exactness import (
    collapsed_rts_smoother,
    exact_path_smoother,
    imm_pair_bank,
    k2_embedding,
    memoryless_c0_model,
    onestep_collapsed_smoother,
    pair_rts_smoother,
    random_cross_annihilation_params,
    slaved_init,
)
from prg.experiments.study import _simulate

N = 6
TOL = 1e-12


def _record(params, seed=0):
    _, _, ys = _simulate(params, N, seed=SEED0 + seed)
    return ys


@pytest.mark.parametrize("make", [lambda: ab_model(0.4), cgo_memory_model])
def test_generic_gpb2_pair_rts_is_the_s7_pair_rts(make):
    p = make()
    ys = _record(p)
    Ex, _ = collapsed_rts_smoother(p, ys, "gpb2", "pair")
    assert _rel(Ex, pair_rts_smoother(p, ys)) < TOL


@pytest.mark.parametrize("bank", ["gpb2", "imm"])
def test_kim_weights_exact_on_the_intersection(bank):
    p = memoryless_c0_model()
    for sd in range(2):
        ys = _record(p, sd)
        Ex_s, _, gam_ex = exact_path_smoother(p, ys)
        Ex, gam = collapsed_rts_smoother(p, ys, bank, "kim")
        assert _rel(Ex, Ex_s) < TOL
        assert _rel(gam, gam_ex) < TOL


@pytest.mark.parametrize("bank", ["gpb2", "imm"])
def test_kim_weights_on_c0_with_memory_regimes_exact_state_biased(bank):
    p = cgo_memory_model()
    ys = _record(p)
    Ex_s, _, gam_ex = exact_path_smoother(p, ys)
    Ex, gam = collapsed_rts_smoother(p, ys, bank, "kim")
    assert _rel(gam, gam_ex) < TOL
    assert _rel(Ex, Ex_s) > 1e-3


def test_on_ab_kim_is_biased_and_gpb2_pair_rts_exact():
    p = ab_model(0.4)
    ys = _record(p)
    Ex_s, _, gam_ex = exact_path_smoother(p, ys)
    Ex_k, gam_k = collapsed_rts_smoother(p, ys, "gpb2", "kim")
    assert _rel(gam_k, gam_ex) > 1e-3
    assert _rel(Ex_k, Ex_s) > 1e-3
    Ex_p, gam_p = collapsed_rts_smoother(p, ys, "gpb2", "pair")
    assert _rel(Ex_p, Ex_s) < TOL
    assert _rel(gam_p, gam_ex) < TOL


def test_k2_embedding_transports_the_filter_not_the_smoother():
    p = slaved_init(ab_model(0.4))
    p2 = k2_embedding(p)
    ys = _record(p)
    Ex_f, _, _ = exact_mixture_filter(p, ys)
    zs2, _, al2, _ = imm_pair_bank(p2, ys)
    assert _rel(np.einsum("nk,nkq->nq", al2, zs2[:, :, : p.q, 0]), Ex_f) < TOL
    Ex_s, _, gam = exact_path_smoother(p, ys)
    Ex_s2, _, gam2 = exact_path_smoother(p2, ys)
    assert _rel(gam2.reshape(-1, p.K, p.K).sum(axis=1), gam) < TOL
    assert _rel(collapsed_rts_smoother(p2, ys, "imm", "kim")[0], Ex_s2) < TOL
    assert _rel(Ex_s2, Ex_s) > 0.1


@pytest.mark.parametrize(
    "make, bank",
    [
        (lambda: ab_model(0.4), "gpb2"),
        (cgo_memory_model, "gpb2"),
        (lambda: cross_annihilation_model(annihilate=True), "gpb2"),
        (cgo_memory_model, "imm"),
    ],
)
def test_onestep_smoother_exact_on_the_domain_of_its_filter(make, bank):
    p = make()
    ys = _record(p)
    Ex_s, Vx_s, gam_ex = exact_path_smoother(p, ys)
    Ex, Vx, gam = onestep_collapsed_smoother(p, ys, bank)
    assert _rel(Ex, Ex_s) < TOL
    assert _rel(Vx, Vx_s) < TOL
    assert _rel(gam, gam_ex) < TOL


def test_onestep_smoother_exact_on_random_cross_annihilation_model():
    p = random_cross_annihilation_params(np.random.default_rng(3), 2, 3, 1)
    ys = _record(p)
    Ex_s, _, _ = exact_path_smoother(p, ys)
    assert _rel(onestep_collapsed_smoother(p, ys, "gpb2")[0], Ex_s) < TOL
    assert _rel(onestep_collapsed_smoother(p, ys, "imm")[0], Ex_s) > 1e-4


def test_onestep_smoother_biased_off_the_gpb2_domain():
    p = cross_annihilation_model(annihilate=False)
    ys = _record(p)
    Ex_s, _, _ = exact_path_smoother(p, ys)
    assert _rel(onestep_collapsed_smoother(p, ys, "gpb2")[0], Ex_s) > 1e-3
