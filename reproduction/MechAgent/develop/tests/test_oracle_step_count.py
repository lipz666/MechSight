from __future__ import annotations

import unittest

from develop.prompts import (
    create_plan_messages,
    directive_messages,
    executor_messages,
    replan_messages,
    review_messages,
)
from develop.schemas import ExecutorProposal, GlobalPlan, MechanismTaskContext, StepDirective


def task(oracle=None):
    return MechanismTaskContext(
        reaction_id="structures_only::7",
        task_description="infer the mechanism",
        initial_state_pgl="ReactionState(initial)",
        target_state_pgl="ReactionState(target)",
        oracle_step_count=oracle,
    )


def plan():
    return GlobalPlan(
        plan_version=1,
        reaction_family="test",
        reaction_summary="test",
        global_mechanism="test",
        elementary_steps=[{"step_id": "E1", "description": "one event"}],
        initial_elementary_step_id="E1",
    )


def directive():
    built = plan()
    return StepDirective(
        directive_id="d1",
        plan_version=1,
        mechanism_step_idx=0,
        global_context=built.global_context(),
        directive_decision="EXECUTE",
        active_elementary_step={"step_id": "E1", "description": "one event"},
        current_position="initial",
        expected_electron_transfer_event="move a lone pair",
    )


def text(messages):
    return "\n".join(message["content"] for message in messages)


class OracleStepCountTests(unittest.TestCase):
    def test_absent_by_default(self):
        rendered = text(create_plan_messages(task()))
        self.assertNotIn("reference_step_count", rendered)
        self.assertNotIn("Step-count reference", rendered)

    def test_planner_receives_the_count_and_its_contract(self):
        rendered = text(create_plan_messages(task(oracle=5)))
        self.assertIn("<reference_step_count>\n5\n</reference_step_count>", rendered)
        self.assertIn("Step-count reference", rendered)
        self.assertIn("Never invent a chemical event to pad the count", rendered)

    def test_replan_also_receives_it(self):
        rendered = text(
            replan_messages(task(oracle=4), plan(), "ReactionState(current)", [], "reason")
        )
        self.assertIn("<reference_step_count>\n4\n</reference_step_count>", rendered)

    def test_the_count_never_reaches_the_executor(self):
        """The ablation constrains planning granularity, not action generation."""

        rendered = text(
            executor_messages(
                task(oracle=5), plan(), directive(), "ReactionState(current)", [], [], None
            )
        )
        self.assertNotIn("reference_step_count", rendered)
        self.assertNotIn("Step-count reference", rendered)

    def test_the_count_never_reaches_the_directive_or_review(self):
        rendered = text(
            directive_messages(
                task(oracle=5),
                plan(),
                {"step_id": "E1", "description": "one event"},
                0,
                "ReactionState(current)",
                [],
                None,
            )
        )
        self.assertNotIn("reference_step_count", rendered)

        proposal = ExecutorProposal(
            proposal_id="p1",
            directive_id="d1",
            review_round=1,
            event_summary="summary",
            plan_objection="",
            action_pgl="ReactionTransformation()",
        )
        rendered = text(
            review_messages(
                task(oracle=5),
                plan(),
                directive(),
                "ReactionState(current)",
                [],
                proposal,
                {"valid": True},
                [],
            )
        )
        self.assertNotIn("reference_step_count", rendered)

    def test_annotated_actions_themselves_never_appear(self):
        rendered = text(create_plan_messages(task(oracle=5)))
        self.assertNotIn("ElectronTransfer", rendered.replace("ElectronTransfer(source", ""))


if __name__ == "__main__":
    unittest.main()
