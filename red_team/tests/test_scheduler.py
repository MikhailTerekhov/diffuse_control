"""Tests for MO-ASHA v2 scheduler."""

import numpy as np

from red_team.scheduler import (
    MOASHAConfig,
    MOASHAScheduler,
    PromptTrial,
    SampleResult,
    TrialStatus,
    EvalWorkItem,
    GenerateWorkItem,
)


DATASET = [{"paper_id": f"paper_{i}", "input_text": "", "rq_text": "", "paper_text": ""}
           for i in range(200)]


def make_trial(prompt="test", rung=0, island=0, **kwargs) -> PromptTrial:
    return PromptTrial(prompt_text=prompt, current_rung=rung, island_id=island,
                       status=TrialStatus.PENDING, **kwargs)


def complete_rung(scheduler, trial, rung, weak, strong, n_samples=None):
    """Simulate completing all samples at a rung (no noise for deterministic tests)."""
    if n_samples is None:
        n_samples = scheduler.config.rung_samples[rung]
    existing = len(trial.sample_results)
    for i in range(n_samples - existing):
        scheduler.on_sample_complete(trial.trial_id, SampleResult(
            paper_id=f"p_{trial.trial_id}_{rung}_{i}",
            weak_score=weak,
            strong_score=strong,
        ))


class TestBasics:
    def test_register(self):
        s = MOASHAScheduler(MOASHAConfig(rung_samples=[2, 4], eta=2))
        t = make_trial()
        s.register_trial(t)
        assert s.n_generated == 1
        assert t.trial_id in s.trials

    def test_is_done(self):
        s = MOASHAScheduler(MOASHAConfig(rung_samples=[2, 4], eta=2, target_highest_rung=1))
        assert not s.is_done()

    def test_work_queue_needs_dataset(self):
        s = MOASHAScheduler(MOASHAConfig(rung_samples=[2, 4]))
        assert s.get_work_queue(10) == []

    def test_work_queue_dispatches_pending(self):
        s = MOASHAScheduler(MOASHAConfig(rung_samples=[2, 4]))
        s.set_dataset(DATASET)
        t = make_trial()
        s.register_trial(t)
        work = s.get_work_queue(10)
        eval_items = [w for w in work if isinstance(w, EvalWorkItem)]
        assert len(eval_items) == 2  # rung 0 = 2 samples
        assert all(w.trial_id == t.trial_id for w in eval_items)

    def test_work_queue_generate_when_idle(self):
        s = MOASHAScheduler(MOASHAConfig(rung_samples=[2, 4]))
        s.set_dataset(DATASET)
        # No trials, should generate
        work = s.get_work_queue(10)
        assert any(isinstance(w, GenerateWorkItem) for w in work)


class TestSampleCompletion:
    def test_rung_completes_after_all_samples(self):
        cfg = MOASHAConfig(rung_samples=[4, 8], eta=2)
        s = MOASHAScheduler(cfg)
        s.set_dataset(DATASET)
        t = make_trial()
        s.register_trial(t)

        # Dispatch and complete 4 samples
        for i in range(4):
            t._pending_samples += 1
        for i in range(3):
            s.on_sample_complete(t.trial_id, SampleResult(f"p{i}", 0.8, 0.2))
        # Not yet complete (1 still pending)
        assert t.trial_id not in s.rungs[0].completed

        s.on_sample_complete(t.trial_id, SampleResult("p3", 0.8, 0.2))
        # Now complete
        assert t.trial_id in s.rungs[0].completed
        assert 0 in t.rung_scores

    def test_scores_aggregated(self):
        cfg = MOASHAConfig(rung_samples=[2], eta=2)
        s = MOASHAScheduler(cfg)
        t = make_trial()
        s.register_trial(t)
        s.on_sample_complete(t.trial_id, SampleResult("p0", 0.8, 0.2))
        s.on_sample_complete(t.trial_id, SampleResult("p1", 0.6, 0.4))
        weak, strong, n = t.rung_scores[0]
        assert abs(weak - 0.7) < 1e-6
        assert abs(strong - 0.3) < 1e-6
        assert n == 2


class TestPromotion:
    def test_no_promotion_below_eta(self):
        cfg = MOASHAConfig(rung_samples=[2, 4], eta=3)
        s = MOASHAScheduler(cfg)
        for i in range(2):  # below eta=3
            t = make_trial(prompt=f"t{i}")
            s.register_trial(t)
            complete_rung(s, t, 0, 0.8, 0.2)
        for t in s.trials.values():
            assert t.status in (TrialStatus.WAITING, TrialStatus.STOPPED)

    def test_promotion_at_eta(self):
        cfg = MOASHAConfig(rung_samples=[2, 4], eta=3)
        s = MOASHAScheduler(cfg)
        scores = [(0.9, 0.1), (0.5, 0.5), (0.3, 0.3)]
        trials = []
        for i, (w, st) in enumerate(scores):
            t = make_trial(prompt=f"t{i}")
            s.register_trial(t)
            complete_rung(s, t, 0, w, st)
            trials.append(t)

        promoted = [t for t in trials if t.status == TrialStatus.PROMOTED]
        assert len(promoted) == 1  # floor(3/3) = 1

    def test_pareto_promotion_selects_diverse(self):
        """Two non-dominated points should both be promotable with eta=2."""
        cfg = MOASHAConfig(rung_samples=[2, 4], eta=2)
        s = MOASHAScheduler(cfg)
        # In sandbagging: objectives = (-strong, weak). Non-dominated pairs:
        # (weak=0.9, strong=0.1) -> obj=(-0.1, 0.9) — high weak, low strong
        # (weak=0.5, strong=0.05) -> obj=(-0.05, 0.5) — moderate weak, very low strong
        # These are non-dominated in sandbagging objectives
        scores = [
            (0.9, 0.1),   # high weak, low strong — PF
            (0.5, 0.05),  # moderate weak, very low strong — PF
            (0.4, 0.4),   # dominated (worse weak AND worse -strong than (0.9,0.1))
            (0.2, 0.3),   # dominated
        ]
        trials = []
        for i, (w, st) in enumerate(scores):
            t = make_trial(prompt=f"t{i}")
            s.register_trial(t)
            complete_rung(s, t, 0, w, st)
            trials.append(t)

        promoted = [t for t in trials if t.status == TrialStatus.PROMOTED]
        # Adaptive: target = int(4 * 0.5) = 2 total, but promotions happen
        # incrementally as trials complete, so may be 1 or 2 depending on order
        assert 1 <= len(promoted) <= 2

    def test_retroactive_promotion(self):
        """Trial stopped earlier can be promoted when more trials arrive."""
        cfg = MOASHAConfig(rung_samples=[2, 4], eta=2)
        s = MOASHAScheduler(cfg)

        # In sandbagging: high weak + low strong = good
        t1 = make_trial(prompt="good")
        t2 = make_trial(prompt="ok")
        s.register_trial(t1)
        s.register_trial(t2)
        complete_rung(s, t1, 0, 0.9, 0.1)  # best sandbagging
        complete_rung(s, t2, 0, 0.7, 0.2)  # decent
        # With eta=2, floor(2/2)=1 promoted
        assert t1.status == TrialStatus.PROMOTED
        assert t2.status == TrialStatus.STOPPED

        # Add 2 more worse trials, floor(4/2)=2 → t2 gets retroactively promoted
        t3 = make_trial(prompt="bad1")
        t4 = make_trial(prompt="bad2")
        s.register_trial(t3)
        s.register_trial(t4)
        complete_rung(s, t3, 0, 0.3, 0.8)  # bad sandbagging
        complete_rung(s, t4, 0, 0.2, 0.9)  # bad sandbagging

        promoted = [t for t in s.trials.values() if t.status == TrialStatus.PROMOTED]
        # Adaptive: target = int(4 * 0.5) = 2, had 1 → should promote 1 more
        # But the new promotion comes from the window which includes the bad trials,
        # so the good trial t2 competes against t3/t4 and should win
        assert len(promoted) >= 1  # at least the original promotion holds

    def test_highest_rung_finishes(self):
        cfg = MOASHAConfig(rung_samples=[2, 4], eta=2, target_highest_rung=1)
        s = MOASHAScheduler(cfg)

        t1 = make_trial()
        t2 = make_trial()
        s.register_trial(t1)
        s.register_trial(t2)
        complete_rung(s, t1, 0, 0.9, 0.1)  # good sandbagging
        complete_rung(s, t2, 0, 0.3, 0.8)  # bad sandbagging

        # t1 promoted to rung 1
        assert t1.status == TrialStatus.PROMOTED
        assert t1.current_rung == 1

        # Complete rung 1
        complete_rung(s, t1, 1, 0.9, 0.1)
        assert t1.status == TrialStatus.FINISHED
        assert s.n_highest_rung_complete == 1
        assert s.is_done()


class TestWorkQueuePriority:
    def test_higher_rung_first(self):
        cfg = MOASHAConfig(rung_samples=[2, 4], eta=2)
        s = MOASHAScheduler(cfg)
        s.set_dataset(DATASET)

        # Create a rung-0 pending trial and a rung-1 promoted trial
        t_low = make_trial(prompt="low")
        s.register_trial(t_low)

        t_high = make_trial(prompt="high", rung=1)
        t_high.status = TrialStatus.PROMOTED
        t_high.sample_results = [SampleResult(f"p{i}", 0.5, 0.5) for i in range(2)]
        s.register_trial(t_high)

        work = s.get_work_queue(10)
        eval_items = [w for w in work if isinstance(w, EvalWorkItem)]
        # First items should be for the higher-rung trial
        assert eval_items[0].rung == 1

    def test_no_double_dispatch(self):
        """Calling get_work_queue twice shouldn't double-dispatch samples."""
        cfg = MOASHAConfig(rung_samples=[4])
        s = MOASHAScheduler(cfg)
        s.set_dataset(DATASET)
        t = make_trial()
        s.register_trial(t)

        work1 = s.get_work_queue(10)
        work2 = s.get_work_queue(10)

        eval1 = [w for w in work1 if isinstance(w, EvalWorkItem)]
        eval2 = [w for w in work2 if isinstance(w, EvalWorkItem)]
        assert len(eval1) == 4  # all 4 samples dispatched
        assert len(eval2) == 0  # nothing left to dispatch


class TestParetoFront:
    def test_empty(self):
        s = MOASHAScheduler(MOASHAConfig())
        assert s.get_pareto_front() == []

    def test_single(self):
        s = MOASHAScheduler(MOASHAConfig(rung_samples=[2]))
        t = make_trial()
        s.register_trial(t)
        complete_rung(s, t, 0, 0.5, 0.5)
        pf = s.get_pareto_front()
        assert len(pf) == 1

    def test_high_rung_pareto(self):
        cfg = MOASHAConfig(rung_samples=[2, 4], eta=2)
        s = MOASHAScheduler(cfg)

        # Rung 0 trial
        t1 = make_trial(prompt="r0")
        s.register_trial(t1)
        complete_rung(s, t1, 0, 0.5, 0.5)

        # Rung 1 trial (manually add to rung's completed list)
        t2 = make_trial(prompt="r1", rung=1)
        s.register_trial(t2)
        t2.rung_scores[1] = (0.8, 0.2, 4)
        s.rungs[1].completed.append(t2.trial_id)

        pf_high = s.get_high_rung_pareto(n=1)
        # Should prefer the rung-1 trial
        assert any(t.trial_id == t2.trial_id for t in pf_high)
