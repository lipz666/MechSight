from __future__ import annotations

import unittest

from develop.orchestrator import PlannerExecutorAgent, PlannerExecutorConfig
from develop.schemas import (
    CompletionReview,
    ExecutorProposal,
    GlobalPlan,
    MechanismTaskContext,
    PlannerReview,
    StepDirective,
)


class _FakeEnv:
    def __init__(self):
        self.current_state = "S0"

    def reset(self):
        self.current_state = "S0"
        return "S0", {"legal_local_actions": [{"source_type": "atom", "target_type": "atom"}]}

    def legal_local_actions(self):
        return [{"source_type": "atom", "target_type": "atom"}]

    def product_reached(self):
        return self.current_state == "TARGET"

    def state_signature(self):
        return {"state": self.current_state}

    def step(self, action):
        self.current_state = "TARGET"
        return "TARGET", 10.0, True, {
            "edit_valid": True,
            "edit_operations": ["applied fake transfer"],
            "edit_error": None,
        }


class _RejectOnceEnv(_FakeEnv):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def step(self, action):
        self.calls += 1
        if self.calls == 1:
            return "S0", -1.0, False, {
                "edit_valid": False,
                "edit_operations": [],
                "edit_error": "synthetic environment rejection",
            }
        return super().step(action)


class _TwoStepEnv(_FakeEnv):
    def step(self, action):
        if self.current_state == "S0":
            self.current_state = "S1"
            return "S1", 1.0, False, {
                "edit_valid": True,
                "edit_operations": ["completed E1"],
                "edit_error": None,
            }
        self.current_state = "TARGET"
        return "TARGET", 1.0, True, {
            "edit_valid": True,
            "edit_operations": ["completed E2"],
            "edit_error": None,
        }


class _BoundaryDifferenceEnv(_FakeEnv):
    def step(self, action):
        self.current_state = "TARGET_PLUS_H2"
        return self.current_state, 1.0, False, {
            "edit_valid": True,
            "edit_operations": ["formed product and H2"],
            "edit_error": None,
        }


class _FakePlanner:
    def __init__(self):
        self.review_count = 0
        self.directive_feedback = []
        self.directive_steps = []

    def create_plan(self, task, *, required_changes=None):
        return GlobalPlan(
            plan_version=1,
            reaction_family="test",
            reaction_summary="test reaction",
            global_mechanism="perform one event",
            elementary_steps=[{"step_id": "E1", "description": "Perform one event."}],
            initial_elementary_step_id="E1",
        )

    def make_directive(
        self,
        *,
        plan,
        mechanism_step_idx,
        directive_id,
        last_env_feedback=None,
        **kwargs,
    ):
        self.directive_feedback.append(last_env_feedback)
        self.directive_steps.append(kwargs["active_elementary_step"]["step_id"])
        return StepDirective(
            directive_id=directive_id,
            plan_version=plan.plan_version,
            mechanism_step_idx=mechanism_step_idx,
            global_context=plan.global_context(),
            directive_decision="EXECUTE",
            active_elementary_step=kwargs["active_elementary_step"],
            current_position="initial",
            expected_electron_transfer_event="perform the fake transfer",
        )

    def review(self, *, proposal, review_id, **kwargs):
        self.review_count += 1
        decision = "REVISE" if self.review_count == 1 else "APPROVE"
        return PlannerReview(
            review_id=review_id,
            proposal_id=proposal.proposal_id,
            decision=decision,
            global_consistency=decision == "APPROVE",
            elementary_step_consistency=True,
            electron_event_consistency=decision == "APPROVE",
            chemical_assessment="correct it" if decision == "REVISE" else "approved",
            required_correction="retry" if decision == "REVISE" else "",
        )

    def replan(self, **kwargs):
        raise AssertionError("replan should not be called")

    def check_completion(self, *, check_id, mechanism_step_idx, **kwargs):
        return CompletionReview(
            check_id=check_id,
            mechanism_step_idx=mechanism_step_idx,
            decision="CONFIRM_COMPLETE",
            atom_connectivity_match=True,
            bond_order_match=True,
            formal_charge_match=True,
            explicit_h_match=True,
            chemical_assessment="strict target confirmed",
        )


class _TwoStepPlanner(_FakePlanner):
    def create_plan(self, task, *, required_changes=None):
        return GlobalPlan(
            plan_version=1,
            reaction_family="test",
            reaction_summary="two-step reaction",
            global_mechanism="perform two elementary events",
            elementary_steps=[
                {"step_id": "E1", "description": "Perform the first event."},
                {"step_id": "E2", "description": "Perform the second event."},
            ],
            initial_elementary_step_id="E1",
        )


class _BoundaryDifferencePlanner(_FakePlanner):
    def check_completion(
        self,
        *,
        check_id,
        mechanism_step_idx,
        strict_product_match,
        plan_exhausted,
        **kwargs,
    ):
        self.asserted_boundary_inputs = (strict_product_match, plan_exhausted)
        return CompletionReview(
            check_id=check_id,
            mechanism_step_idx=mechanism_step_idx,
            decision="CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE",
            atom_connectivity_match=False,
            bond_order_match=True,
            formal_charge_match=True,
            explicit_h_match=False,
            chemical_assessment="principal product is complete; target omits H2",
            ignored_boundary_differences=["H2 byproduct omitted from target"],
        )


class _UnfinishedProductEnv(_FakeEnv):
    """Ends with a bond still missing from the principal product."""

    def step(self, action):
        self.current_state = "S1"
        return self.current_state, 1.0, False, {
            "edit_valid": True,
            "edit_operations": ["partial edit"],
            "edit_error": None,
        }


def _signature_of(state):
    """Minimal strict signature: the product bond only exists in TARGET states."""

    atoms = {name: {"element": name[0], "Hcount": 0, "charge": 0} for name in ("C1", "O2")}
    bonds = {}
    if str(state).startswith("TARGET"):
        bonds["C1|O2"] = {"kind": "single", "order": 1.0}
    return {"atoms": atoms, "bonds": bonds}


class _FakeExecutor:
    def __init__(self):
        self.feedback = []

    def propose(self, *, proposal_id, directive, review_round, planner_feedback, **kwargs):
        self.feedback.append(planner_feedback)
        return ExecutorProposal(
            proposal_id=proposal_id,
            directive_id=directive.directive_id,
            review_round=review_round,
            event_summary="fake transfer",
            plan_objection="",
            action_pgl="GOOD_ACTION",
        )


class OrchestratorTests(unittest.TestCase):
    @staticmethod
    def _task():
        return MechanismTaskContext(
            reaction_id="fake",
            task_description="fake task",
            initial_state_pgl="S0",
            target_state_pgl="TARGET",
        )

    def test_revision_then_approval_executes_only_approved_action(self):
        planner = _FakePlanner()
        executor = _FakeExecutor()
        env = _FakeEnv()
        runtime = PlannerExecutorAgent(
            env=env,
            planner=planner,
            executor=executor,
            state_to_pgl=lambda state: str(state),
            parse_action=lambda text: ({"action": text}, {"valid": True}),
            validate_action=lambda action, state, legal: {"valid": True, "errors": []},
            config=PlannerExecutorConfig(max_steps=2, max_env_calls=2, max_planner_review_rounds=3),
        )
        result = runtime.run(self._task())
        self.assertTrue(result.success)
        self.assertEqual(result.state.control.env_call_count, 1)
        self.assertEqual(result.state.control.mechanism_step_idx, 1)
        self.assertEqual(result.state.control.plan_step_idx, 1)
        self.assertIsNone(result.state.active_elementary_step_id)
        self.assertEqual(len(result.planner_reviews), 2)
        self.assertEqual(len(result.completion_reviews), 1)
        self.assertIsNone(executor.feedback[0])
        self.assertEqual(executor.feedback[1]["decision"], "REVISE")
        self.assertEqual(result.state.environment.history_actions_pgl, ["GOOD_ACTION"])

    def test_environment_rejection_returns_feedback_without_false_loop_stop(self):
        env = _RejectOnceEnv()
        planner = _FakePlanner()
        runtime = PlannerExecutorAgent(
            env=env,
            planner=planner,
            executor=_FakeExecutor(),
            state_to_pgl=lambda state: str(state),
            parse_action=lambda text: ({"action": text}, {"valid": True}),
            validate_action=lambda action, state, legal: {"valid": True, "errors": []},
            config=PlannerExecutorConfig(max_steps=2, max_env_calls=2, max_planner_review_rounds=3),
        )
        result = runtime.run(self._task())
        self.assertTrue(result.success)
        self.assertEqual(result.state.control.env_call_count, 2)
        self.assertEqual([item["edit_valid"] for item in result.execution_observations], [False, True])
        self.assertEqual(result.state.environment.history_actions_pgl, ["GOOD_ACTION"])
        self.assertNotIn("reward", planner.directive_feedback[1])
        self.assertEqual(planner.directive_steps, ["E1", "E1"])

    def test_plan_exhaustion_can_finish_with_annotation_boundary_difference(self):
        planner = _BoundaryDifferencePlanner()
        runtime = PlannerExecutorAgent(
            env=_BoundaryDifferenceEnv(),
            planner=planner,
            executor=_FakeExecutor(),
            state_to_pgl=lambda state: str(state),
            parse_action=lambda text: ({"action": text}, {"valid": True}),
            validate_action=lambda action, state, legal: {"valid": True, "errors": []},
            config=PlannerExecutorConfig(max_steps=2, max_env_calls=2),
        )
        result = runtime.run(self._task())
        self.assertTrue(result.success)
        self.assertEqual(
            result.state.control.stop_reason,
            "mechanism_complete_with_boundary_difference",
        )
        self.assertEqual(planner.asserted_boundary_inputs, (False, True))
        self.assertEqual(
            result.completion_reviews[0]["ignored_boundary_differences"],
            ["H2 byproduct omitted from target"],
        )

    def test_boundary_completion_is_refused_while_the_product_is_unfinished(self):
        """A boundary difference is an omitted species, not a missing bond.

        Without the gate the Planner's own CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE
        ends the rollout unchallenged, which is how 156 of 180 Gemini rollouts
        terminated against 24 strict structural matches.
        """

        env = _UnfinishedProductEnv()
        planner = _BoundaryDifferencePlanner()
        runtime = PlannerExecutorAgent(
            env=env,
            planner=planner,
            executor=_FakeExecutor(),
            state_to_pgl=lambda state: str(state),
            parse_action=lambda text: ({"action": text}, {"valid": True}),
            validate_action=lambda action, state, legal: {"valid": True, "errors": []},
            config=PlannerExecutorConfig(max_steps=2, max_env_calls=2),
            state_signature=_signature_of,
        )
        env.final_state = "TARGET"
        result = runtime.run(self._task())

        self.assertFalse(result.success)
        self.assertNotEqual(
            result.state.control.stop_reason,
            "mechanism_complete_with_boundary_difference",
        )
        override = result.completion_reviews[0]["runtime_override"]
        self.assertEqual(
            override["original_decision"], "CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE"
        )
        self.assertEqual(override["enforced_decision"], "CONTINUE")

    def test_boundary_completion_still_allowed_for_an_omitted_byproduct(self):
        env = _BoundaryDifferenceEnv()
        planner = _BoundaryDifferencePlanner()
        runtime = PlannerExecutorAgent(
            env=env,
            planner=planner,
            executor=_FakeExecutor(),
            state_to_pgl=lambda state: str(state),
            parse_action=lambda text: ({"action": text}, {"valid": True}),
            validate_action=lambda action, state, legal: {"valid": True, "errors": []},
            config=PlannerExecutorConfig(max_steps=2, max_env_calls=2),
            state_signature=_signature_of,
        )
        env.final_state = "TARGET"
        result = runtime.run(self._task())

        self.assertTrue(result.success)
        self.assertEqual(
            result.state.control.stop_reason,
            "mechanism_complete_with_boundary_difference",
        )
        self.assertNotIn("runtime_override", result.completion_reviews[0])

    def test_successful_action_advances_exactly_one_elementary_step(self):
        env = _TwoStepEnv()
        planner = _TwoStepPlanner()
        runtime = PlannerExecutorAgent(
            env=env,
            planner=planner,
            executor=_FakeExecutor(),
            state_to_pgl=lambda state: str(state),
            parse_action=lambda text: ({"action": text}, {"valid": True}),
            validate_action=lambda action, state, legal: {"valid": True, "errors": []},
            config=PlannerExecutorConfig(max_steps=3, max_env_calls=3, max_planner_review_rounds=3),
        )
        result = runtime.run(self._task())
        self.assertTrue(result.success)
        self.assertEqual(planner.directive_steps, ["E1", "E2"])
        self.assertEqual(
            [item["elementary_step_id"] for item in result.execution_observations],
            ["E1", "E2"],
        )
        self.assertEqual(result.state.control.mechanism_step_idx, 2)
        self.assertEqual(result.state.control.plan_step_idx, 2)
        self.assertIsNone(result.state.active_elementary_step_id)

class _AssertsCompleteOnReplanPlanner(_BoundaryDifferencePlanner):
    """Returns no remaining steps, as the model does when it disagrees."""

    def replan(self, **kwargs):
        raise ValueError("elementary_steps must be a non-empty list")


class ReplanDisagreementTests(unittest.TestCase):
    def test_an_empty_replan_is_a_disagreement_not_a_crash(self):
        env = _UnfinishedProductEnv()
        runtime = PlannerExecutorAgent(
            env=env,
            planner=_AssertsCompleteOnReplanPlanner(),
            executor=_FakeExecutor(),
            state_to_pgl=lambda state: str(state),
            parse_action=lambda text: ({"action": text}, {"valid": True}),
            validate_action=lambda action, state, legal: {"valid": True, "errors": []},
            config=PlannerExecutorConfig(max_steps=2, max_env_calls=2),
            state_signature=_signature_of,
        )
        env.final_state = "TARGET"
        result = runtime.run(OrchestratorTests._task())
        self.assertEqual(result.state.control.stop_reason, "replan_asserts_complete")
        self.assertNotIn(
            "planner_replan_error", [error.get("phase") for error in result.errors]
        )

if __name__ == "__main__":
    unittest.main()
