"""Block-level exactness-domain residuals (prg.utils.exactness) against the
gauges of the committed experiment suite and the exact mixture filter."""

from __future__ import annotations

from prg.experiments.cns_exactness import (
    ab_model,
    cgo_memory_model,
    cross_annihilation_model,
    g2_degenerate_model,
    g2_vector_model,
    jmls_cross_annihilation_model,
    mixed_branch_model,
    off_union_model,
    slaving_A_model,
)
from prg.utils.exactness import (
    assumption_g,
    cross_annihilation_residual,
    exactness_domains,
    gpb2_domain_residual,
    imm_domain_residual,
    residual_memory,
)


def test_cgo_model_is_imm_and_gpb2_exact():
    p = cgo_memory_model()
    assert imm_domain_residual(p) == 0.0
    assert gpb2_domain_residual(p) == 0.0
    d = exactness_domains(p)
    assert d["imm"] and d["gpb2"] and not d["constant_gain"]


def test_slaving_only_model_is_gpb2_exact_not_imm():
    p = slaving_A_model(0.4)
    assert imm_domain_residual(p) > 1e-3
    assert gpb2_domain_residual(p) < 1e-12
    d = exactness_domains(p)
    assert d["gpb2"] and not d["imm"] and not d["constant_gain"]


def test_ab_model_all_three_exact_but_imm():
    d = exactness_domains(ab_model(0.4))
    assert d["gpb2"] and d["constant_gain"] and not d["imm"]


def test_round_off_memory_is_on_the_gpb2_domain():
    from prg.classes.GSSParams import GSSParams
    from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1

    p = GSSParams.from_model(ModelGssK2Q1S1())  # AB up to the rounding of its stored blocks
    assert 0.0 < max(abs(residual_memory(p, j)).max() for j in range(p.K)) < 1e-12
    assert gpb2_domain_residual(p) < 1e-12
    d = exactness_domains(p)
    assert d["gpb2"] and d["constant_gain"]


def test_mixed_branch_leaves_gpb2_domain_unless_memoryless():
    assert gpb2_domain_residual(mixed_branch_model(0.0, 0.7)) < 1e-12
    r, (j, k) = cross_annihilation_residual(mixed_branch_model(0.8, 0.7))
    assert r > 1e-3
    assert (j, k) == (0, 1)  # memory of regime 0 seen by the channel of regime 1


def test_off_union_model_is_off_domain():
    assert gpb2_domain_residual(off_union_model(0.4, 0.2)) > 1e-3


def test_matrix_cross_annihilation_beyond_the_union():
    """q=2: singular nonzero memories with C_k N_j = 0 (E10) -- inside the GPB2
    domain although outside {C==0} U {A==MC}; rotated rows leave it."""
    p_in, p_out = cross_annihilation_model(True), cross_annihilation_model(False)
    assert gpb2_domain_residual(p_in) < 1e-12
    assert imm_domain_residual(p_in) > 1e-3
    assert gpb2_domain_residual(p_out) > 1e-2


def test_gpb2_domain_matches_exact_filter():
    """The block-level test predicts machine-precision agreement with the exact
    mixture filter (and disagreement off the domain)."""
    from prg.experiments.cns_exactness import gaps

    g_in = gaps(cross_annihilation_model(True), n_seeds=5)
    g_out = gaps(cross_annihilation_model(False), n_seeds=5)
    assert g_in["gpb2"][0].hi < 1e-12
    assert g_out["gpb2"][0].med > 1e-5


def test_assumption_g_holds_on_the_suite_models():
    for p in (
        cgo_memory_model(),
        slaving_A_model(0.4),
        ab_model(0.4),
        off_union_model(0.4, 0.2),
        mixed_branch_model(0.8, 0.7),
        cross_annihilation_model(True),
    ):
        g = assumption_g(p)
        assert g["g1"] and g["g2"] and g["g"], g
        assert exactness_domains(p)["assumption_g"]


def test_assumption_g2_fails_on_the_e8_family_and_is_restored_by_distinct_rows():
    # E8: regime-free observation row and i.i.d. regime -> (G2) fails
    g = assumption_g(g2_degenerate_model(0.5, distinct_rows=False))
    assert g["g1"] and not g["g2"] and not g["g"]
    assert g["g2_rows"] < 1e-12 and g["g2_channel"] < 1e-12
    # distinct transition rows alone restore (G2)
    g = assumption_g(g2_degenerate_model(0.5, distinct_rows=True))
    assert g["g2"] and g["g2_rows"] > 0.1


def test_assumption_g2_reads_the_state_informed_part_not_the_whole_row():
    # E8 vector case: the observation row is regime-dependent through an
    # autonomous component (D_22), yet its state-informed part is regime-free
    g = assumption_g(g2_vector_model(couple=False))
    assert not g["g2"], g
    # coupling y2 into y1 through a regime-dependent D_12 makes C^T SV^-1 D
    # regime-dependent and restores (G2)
    g = assumption_g(g2_vector_model(couple=True))
    assert g["g2"] and g["g2_channel"] > 1e-3, g


def test_jmls_in_cross_annihilation_family_is_not_constant_gain_exact():
    """Regression: a JMLS (K=2, q=2, s=1) with A_r - M_r C_r != 0 whose rows
    annihilate every memory. The former pairwise regression test accepted it as
    AB (residual exactly 0) although the constant-gain filter is biased on it;
    the block test rejects it. GPB2 is exact (cross-annihilation)."""
    import numpy as np

    from prg.experiments.cns_exactness import _rel, _run_cg
    from prg.experiments.reference_filters import exact_mixture_filter, gpb2_filter
    from prg.experiments.study import _simulate
    from prg.utils.ab_constraint import ab_residual_max, pu_residual_max, validate_ngh_msm

    p = jmls_cross_annihilation_model()
    # the blocks: A_r - M_r C_r = [[0, 0.30], [0, 0]] and [[0, -0.26], [0, 0]]
    for r, off in ((0, 0.3), (1, -0.9 / 3.4)):
        N = p.f_matrix.A(r) - p.noise_cov.M(r) @ p.f_matrix.C(r)
        np.testing.assert_allclose(N, [[0.0, off], [0.0, 0.0]], atol=1e-12)
    assert pu_residual_max(p)[0] < 1e-12  # the regression residual sees nothing
    assert ab_residual_max(p)[0] > 0.1  # the block test does
    assert any("AB constraint violated" in m for m in validate_ngh_msm(p))
    d = exactness_domains(p)
    assert d == {"imm": False, "gpb2": True, "constant_gain": False, "assumption_g": True}

    worst_cg = worst_gpb2 = 0.0
    for seed in range(700, 710):
        _, _, ys = _simulate(p, 10, seed=seed)
        ex_exact = exact_mixture_filter(p, ys)[0]
        worst_cg = max(worst_cg, _rel(_run_cg(p, ys)[0], ex_exact))
        worst_gpb2 = max(worst_gpb2, _rel(gpb2_filter(p, ys)[0], ex_exact))
    assert worst_cg > 1e-4, f"constant-gain filter unexpectedly exact: {worst_cg:.2e}"
    assert worst_gpb2 < 1e-12, f"GPB2 should be exact: {worst_gpb2:.2e}"
