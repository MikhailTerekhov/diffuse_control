"""Tests for PF-gap-targeting in PromptGenerator + scheduler.inject_completed_trial."""

import sys
from pathlib import Path

# Make project importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from red_team.scheduler import (
    MOASHAConfig,
    MOASHAScheduler,
    PromptTrial,
    SampleResult,
    TrialStatus,
)
from red_team.generator import PromptGenerator


def _make_completed_trial(weak: float, strong: float, rung: int = 2,
                          prompt: str = "p", n: int = 10) -> PromptTrial:
    samples = [SampleResult(paper_id=f"p{i}", weak_score=weak, strong_score=strong)
               for i in range(n)]
    t = PromptTrial(
        prompt_text=prompt,
        current_rung=rung,
        sample_results=samples,
        rung_scores={r: (weak, strong, n) for r in range(rung + 1)},
        status=TrialStatus.FINISHED,
    )
    return t


def _make_scheduler(rung_samples=(8, 32, 128)):
    cfg = MOASHAConfig(rung_samples=list(rung_samples), eta=4,
                       mode="sandbagging_detector")
    s = MOASHAScheduler(cfg)
    s.set_dataset([])
    return s


def test_inject_completed_trial_appears_at_max_rung():
    s = _make_scheduler()
    t = _make_completed_trial(weak=0.7, strong=0.6, rung=2,
                               prompt="seeded prompt")
    s.inject_completed_trial(t)
    assert t.trial_id in s.trials
    assert t.trial_id in s.rungs[2].completed
    # Not promoted (it's already at max rung)
    assert t.trial_id not in s.rungs[2].promoted
    # Should appear in PF
    pf = s.get_pareto_front()
    assert t.trial_id in {x.trial_id for x in pf}


def test_inject_completed_trial_doesnt_bump_n_generated():
    s = _make_scheduler()
    s.n_generated = 100
    t = _make_completed_trial(weak=0.7, strong=0.6)
    s.inject_completed_trial(t)
    assert s.n_generated == 100  # seed shouldn't count as a fresh generation


def test_generator_gap_section_in_system_message():
    s = _make_scheduler()
    # PF top: high weak but lower strong
    pf_top = _make_completed_trial(weak=0.65, strong=0.35, prompt="pf_top_prompt")
    # Other PF members
    pf_mid = _make_completed_trial(weak=0.50, strong=0.20, prompt="pf_mid_prompt")
    # Seed: max_weak corner — high strong + high weak
    seed = _make_completed_trial(weak=0.70, strong=0.73, prompt="seed_prompt")

    s.register_trial(pf_top)
    s.rungs[2].completed.append(pf_top.trial_id)
    s.register_trial(pf_mid)
    s.rungs[2].completed.append(pf_mid.trial_id)
    s.inject_completed_trial(seed)

    cfg = {
        "moasha": {"mode": "sandbagging_detector", "rung_samples": [8, 32, 128]},
        "eval_models": {"weak_scorer_type": "detector"},
        "optimizer": {"n_inspiration_prompts": 3},
        "pf_gap": {"enabled": True},
    }
    gen = PromptGenerator(cfg)
    gen.set_seeded_ids([seed.trial_id])
    gen._maybe_init_gap(s)

    msg = gen._system_message
    # The section should be present with the actual coordinates
    assert "Target region for this resumed run" in msg
    assert "0.350" in msg or "0.35" in msg  # s_low
    assert "0.730" in msg or "0.73" in msg  # s_high
    # Gap should be initialized
    assert gen._gap_low == (0.35, 0.65)
    assert gen._gap_high == (0.73, 0.70)


def test_generator_inspiration_filtered_to_band():
    """The band is by construction empty in the existing PF (that's why we want
    to fill it). So the filter should exclude all out-of-band PF points and
    only show the two boundary points."""
    s = _make_scheduler()
    pf_top = _make_completed_trial(weak=0.65, strong=0.35, prompt="pf_top_prompt")
    # Below band: lower weak, lower strong — non-dominated, but s<0.35
    pf_below = _make_completed_trial(weak=0.30, strong=0.20, prompt="below_band_prompt")
    seed = _make_completed_trial(weak=0.70, strong=0.73, prompt="seed_prompt")

    s.register_trial(pf_top)
    s.rungs[2].completed.append(pf_top.trial_id)
    s.register_trial(pf_below)
    s.rungs[2].completed.append(pf_below.trial_id)
    s.inject_completed_trial(seed)

    cfg = {
        "moasha": {"mode": "sandbagging_detector", "rung_samples": [8, 32, 128]},
        "eval_models": {"weak_scorer_type": "detector"},
        "pf_gap": {"enabled": True},
    }
    gen = PromptGenerator(cfg)
    gen.set_seeded_ids([seed.trial_id])
    gen._maybe_init_gap(s)

    pareto = s.get_pareto_front()
    user_msg = gen._build_prompt(parent_trial=None, pareto_front=pareto,
                                 island_manager=None, island_id=0, scheduler=s)

    # Boundaries always included
    assert "pf_top_prompt" in user_msg
    assert "seed_prompt" in user_msg
    # Out-of-band PF point (s=0.20 < s_low=0.35) is filtered out
    assert "below_band_prompt" not in user_msg


def test_generator_inspiration_includes_in_band_random():
    """Random/island examples in the band should appear (pre-existing prompts
    that happen to fall in the band even if dominated)."""
    s = _make_scheduler()
    pf_top = _make_completed_trial(weak=0.65, strong=0.35, prompt="pf_top_prompt")
    seed = _make_completed_trial(weak=0.70, strong=0.73, prompt="seed_prompt")
    # An in-band prompt that's dominated by pf_top but appears in island pool
    in_band_dominated = _make_completed_trial(weak=0.50, strong=0.50,
                                               prompt="in_band_prompt")

    s.register_trial(pf_top)
    s.rungs[2].completed.append(pf_top.trial_id)
    s.register_trial(in_band_dominated)
    s.rungs[2].completed.append(in_band_dominated.trial_id)
    s.inject_completed_trial(seed)

    # Build a stub island manager with all three trials in island 0's pool
    from red_team.islands import IslandConfig, IslandManager
    icfg = IslandConfig(n_islands=1, mode="sandbagging_detector")
    islands = IslandManager(icfg, n_rungs=3)
    islands.set_completed_lists(s.rungs)
    for t in (pf_top, in_band_dominated, seed):
        islands.insert(0, 2, t.trial_id, t.objectives[0], t.objectives[1],
                       len(t.prompt_text))

    cfg = {
        "moasha": {"mode": "sandbagging_detector", "rung_samples": [8, 32, 128]},
        "eval_models": {"weak_scorer_type": "detector"},
        "pf_gap": {"enabled": True},
    }
    gen = PromptGenerator(cfg)
    gen.set_seeded_ids([seed.trial_id])
    gen._maybe_init_gap(s)

    user_msg = gen._build_prompt(parent_trial=None,
                                 pareto_front=s.get_pareto_front(),
                                 island_manager=islands, island_id=0, scheduler=s)

    # The dominated-but-in-band prompt should appear in random/MAP-Elites section
    # since strong=0.50 is in the band [0.35, 0.73]
    assert "in_band_prompt" in user_msg


def test_generator_gap_disabled_when_no_seed():
    s = _make_scheduler()
    pf = _make_completed_trial(weak=0.5, strong=0.3)
    s.register_trial(pf)
    s.rungs[2].completed.append(pf.trial_id)

    cfg = {
        "moasha": {"mode": "sandbagging_detector", "rung_samples": [8, 32, 128]},
        "eval_models": {"weak_scorer_type": "detector"},
        "pf_gap": {"enabled": True},
    }
    gen = PromptGenerator(cfg)
    gen.set_seeded_ids([])
    gen._maybe_init_gap(s)

    # No seeded → gap auto-disables
    assert not gen._gap_enabled
    assert "Target region" not in gen._system_message
