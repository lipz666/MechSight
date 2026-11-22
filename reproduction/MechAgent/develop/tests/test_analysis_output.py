from __future__ import annotations

import unittest

from develop.run_agent import build_analysis_record
from develop.schemas import (
    ControlState,
    EnvironmentState,
    GlobalPlan,
    MechanismTaskContext,
    PlannerExecutorResult,
    PlannerExecutorState,
)


class AnalysisOutputTests(unittest.TestCase):
    def test_compact_record_contains_gt_plan_and_proposal_status(self):
        task = MechanismTaskContext(
            reaction_id="rxn",
            task_description="infer mechanism",
            initial_state_pgl="S0",
            target_state_pgl="S2",
            conditions="base",
        )
        plan = GlobalPlan(
            plan_version=1,
            reaction_family="SN2",
            reaction_summary="substitution",
            global_mechanism="activate then substitute",
            elementary_steps=[
                {"step_id": "E1", "description": "activate"},
                {"step_id": "E2", "description": "substitute"},
            ],
            initial_elementary_step_id="E1",
        )
        state = PlannerExecutorState(
            task=task,
            plan=plan,
            active_elementary_step_id=None,
            environment=EnvironmentState(
                current_state_pgl="S2",
                history_states_pgl=["S0", "S2"],
                history_actions_pgl=["ACTION_1"],
            ),
            control=ControlState(done=True, success=True, stop_reason="product_reached"),
        )
        result = PlannerExecutorResult(
            state=state,
            plan_versions=[plan.to_dict()],
            executor_proposals=[
                {
                    "proposal_id": "p1",
                    "directive_id": "d1",
                    "review_round": 0,
                    "event_summary": "attack",
                    "plan_objection": "",
                    "action_pgl": "ACTION_1",
                    "validation_result": {"valid": True},
                },
                {
                    "proposal_id": "p2",
                    "directive_id": "d1",
                    "review_round": 1,
                    "event_summary": "bad attack",
                    "plan_objection": "",
                    "action_pgl": "BAD_ACTION",
                    "validation_result": {"valid": False},
                },
            ],
            planner_reviews=[
                {
                    "review_id": "r1",
                    "proposal_id": "p1",
                    "decision": "APPROVE",
                    "chemical_assessment": "correct",
                }
            ],
            execution_observations=[
                {
                    "approved_review_id": "r1",
                    "edit_valid": True,
                    "repeated_state": False,
                }
            ],
        )

        record = build_analysis_record(
            result=result,
            source_reaction_id="mechfinder::SN2::1",
            task_index=1,
            repeat_index=2,
            repeats_per_task=3,
            task_input_mode="metadata",
            gold_actions_pgl=["GT_1", "GT_2"],
            evaluator_assessment={
                "planner_global_mechanism_judgment": "correct",
                "planner_elementary_steps_judgment": "partially_correct",
                "predicted_events_judgment": "partially_correct",
            },
        )

        self.assertEqual(
            set(record), {"reaction", "gt", "planner_plan", "actions"}
        )
        self.assertEqual(record["gt"]["actions_pgl"], ["GT_1", "GT_2"])
        self.assertEqual(
            record["gt"]["evaluator_assessment"][
                "planner_global_mechanism_judgment"
            ],
            "correct",
        )
        self.assertEqual(record["planner_plan"]["reaction_family"], "SN2")
        self.assertTrue(record["actions"][0]["accepted_into_history"])
        self.assertEqual(record["actions"][0]["planner_decision"], "APPROVE")
        self.assertFalse(record["actions"][1]["executed_in_environment"])


if __name__ == "__main__":
    unittest.main()
