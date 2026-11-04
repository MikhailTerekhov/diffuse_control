"""Tests for J(W) — weighted per-paper diagonal MSE (minimization target, J >= 0)."""

import pytest

from blue_team.population import CachedSample, PromptEntry
from blue_team.w_evaluator import compute_J, per_prompt_loss


def make_entry(role: str, h: str, s_mean: float, n_papers: int = 4):
    """Each paper gets the same strong score (= s_mean)."""
    return PromptEntry(
        prompt_hash=h, role=role, prompt_text=f"prompt {h}",
        samples=[CachedSample(paper_id=f"paper_{i}", proposal_path="",
                              s_score=s_mean) for i in range(n_papers)],
    )


def paired_const(p_star, w_per_prompt: dict[str, float]):
    """Helper for legacy-style tests: every paper of prompt ph has w = w_per_prompt[ph]."""
    out = {}
    for p in p_star:
        w = w_per_prompt.get(p.prompt_hash)
        if w is None:
            out[p.prompt_hash] = []
            continue
        out[p.prompt_hash] = [(w, s.s_score) for s in p.samples]
    return out


def test_perfect_diagonal_is_zero():
    p_def = make_entry("p_def", "DEF", s_mean=0.8)
    f0_a = make_entry("F_0", "FA", s_mean=0.2)
    f0_b = make_entry("F_0", "FB", s_mean=0.5)
    p_star = [p_def, f0_a, f0_b]
    paired = paired_const(p_star, {"DEF": 0.8, "FA": 0.2, "FB": 0.5})
    assert compute_J(p_star, paired) == pytest.approx(0.0, abs=1e-12)


def test_anti_diagonal_is_positive():
    p_def = make_entry("p_def", "DEF", s_mean=0.8)
    f0_a = make_entry("F_0", "FA", s_mean=0.2)
    f0_b = make_entry("F_0", "FB", s_mean=0.5)
    p_star = [p_def, f0_a, f0_b]
    paired = paired_const(p_star, {"DEF": 0.2, "FA": 0.9, "FB": 0.5})
    assert compute_J(p_star, paired) > 0.0


def test_group_weights_cycle1():
    a = make_entry("anchor", "A", s_mean=0.7)
    f0 = make_entry("F_0", "F", s_mean=0.2)
    p_star = [a, f0]
    # Anchor perfect, F_0 off by 0.5 (uniformly across all papers)
    paired = paired_const(p_star, {"A": 0.7, "F": 0.7})
    # MSE_anchor = 0; MSE_F_0 = 0.25; weight: anchor=0.5, F_0=0.5
    # J = 0.5*0 + 0.5*0.25 = 0.125
    assert compute_J(p_star, paired) == pytest.approx(0.125)


def test_group_weights_cycle2():
    a = make_entry("anchor", "A", s_mean=0.5)
    f0 = make_entry("F_0", "F0", s_mean=0.5)
    f1 = make_entry("F_1", "F1", s_mean=0.5)
    p_star = [a, f0, f1]
    paired = paired_const(p_star, {"A": 0.8, "F0": 0.8, "F1": 0.8})
    # cycle 2: F_1 (current) = 0.5, anchor = 0.25, F_0 = 0.25
    # all per-paper diffs = 0.3 → MSE = 0.09
    expected = (0.5 + 0.25 + 0.25) * 0.09
    assert compute_J(p_star, paired) == pytest.approx(expected)


def test_per_prompt_loss_sum_equals_J():
    a = make_entry("anchor", "A", s_mean=0.6)
    f0_a = make_entry("F_0", "FA", s_mean=0.3)
    f0_b = make_entry("F_0", "FB", s_mean=0.1)
    p_star = [a, f0_a, f0_b]
    paired = paired_const(p_star, {"A": 0.5, "FA": 0.6, "FB": 0.7})
    J = compute_J(p_star, paired)
    losses = per_prompt_loss(p_star, paired)
    total = sum(c for _, _, c in losses.values())
    assert total == pytest.approx(J, abs=1e-12)


def test_per_prompt_loss_residual_sign():
    a = make_entry("anchor", "A", s_mean=0.6)
    f0 = make_entry("F_0", "F", s_mean=0.2)
    p_star = [a, f0]
    paired = paired_const(p_star, {"A": 0.4, "F": 0.7})   # A under by 0.2; F over by 0.5
    losses = per_prompt_loss(p_star, paired)
    # losses[ph] = (mean_residual, mse_residual, contribution)
    assert losses["A"][0] == pytest.approx(-0.2)
    assert losses["F"][0] == pytest.approx(+0.5)
    # mse residuals are non-negative
    assert losses["A"][1] >= 0
    assert losses["F"][1] >= 0
    # contributions are non-negative
    assert losses["A"][2] >= 0
    assert losses["F"][2] >= 0


def test_paper_level_variance_increases_J():
    """Two prompts with same E[w]=E[s] but different per-paper paired residuals.
    The one with non-constant residual should have a strictly larger J."""
    p_const = PromptEntry(
        prompt_hash="C", role="anchor", prompt_text="",
        samples=[CachedSample(paper_id=f"p{i}", proposal_path="", s_score=0.5)
                 for i in range(4)],
    )
    p_var = PromptEntry(
        prompt_hash="V", role="anchor", prompt_text="",
        samples=[CachedSample(paper_id=f"p{i}", proposal_path="", s_score=0.5)
                 for i in range(4)],
    )
    # Both prompts have mean(w) = 0.5 and mean(s) = 0.5
    # but p_var has paper-level disagreement (+0.3, +0.3, -0.3, -0.3 around 0.5).
    paired_const = {"C": [(0.5, 0.5)] * 4}
    paired_var = {"V": [(0.8, 0.5), (0.8, 0.5), (0.2, 0.5), (0.2, 0.5)]}
    J_const = compute_J([p_const], paired_const)
    J_var = compute_J([p_var], paired_var)
    assert J_const == pytest.approx(0.0)
    assert J_var > 0.0   # Jensen kicks in: E[(w-s)^2]=0.09 even though E[w-s]=0


if __name__ == "__main__":
    for fn in (test_perfect_diagonal_is_zero, test_anti_diagonal_is_positive,
               test_group_weights_cycle1, test_group_weights_cycle2,
               test_per_prompt_loss_sum_equals_J,
               test_per_prompt_loss_residual_sign,
               test_paper_level_variance_increases_J):
        fn()
    print("All tests passed")
