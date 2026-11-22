from __future__ import annotations

import unittest
from types import SimpleNamespace

from develop.agents import PlannerAgent, PosthocGroundTruthEvaluator
from develop.client import OpenAIChatConfig, OpenAIChatModel
from develop.parsers import parse_executor_output, parse_json_tag
from develop.prompts import create_plan_messages, executor_messages
from develop.schemas import (
    GlobalPlan,
    GroundTruthAssessment,
    MechanismTaskContext,
    PlannerReview,
    StepDirective,
)


class _Message:
    def model_dump(self):
        return {
            "reasoning_content": "internal reasoning",
            "content": "<plan>{\"reaction_family\":\"x\"}</plan>",
        }


class _Response:
    class _Choices:
        message = _Message()

    choices = [_Choices()]


class _Completions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _Response()


class _Client:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _Completions()})()


class _SequenceModel:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        self.events = []
        self.message_batches = []

    def complete(self, messages, *, phase, enable_thinking=None):
        self.calls.append(enable_thinking)
        self.message_batches.append(messages)
        self.events.append({"phase": phase})
        return SimpleNamespace(answer=self.answers.pop(0))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.task = MechanismTaskContext(
            reaction_id="rxn-1",
            task_description="Infer the next elementary mechanism action.",
            initial_state_pgl="ReactionState(initial)",
            target_state_pgl="ReactionState(target)",
            conditions="acid",
        )
        self.plan = GlobalPlan(
            plan_version=1,
            reaction_family="test family",
            reaction_summary="test summary",
            global_mechanism="first activate, then form the target bond",
            elementary_steps=[{"step_id": "E1", "description": "Activate the electrophile."}],
            initial_elementary_step_id="E1",
        )
        self.directive = StepDirective(
            directive_id="d1",
            plan_version=1,
            mechanism_step_idx=0,
            global_context=self.plan.global_context(),
            directive_decision="EXECUTE",
            active_elementary_step=self.plan.elementary_step("E1"),
            current_position="At the initial state.",
            expected_electron_transfer_event="Move a lone pair toward the electrophile.",
        )

    def test_json_and_executor_tag_parsing(self):
        plan = parse_json_tag('<plan>{"reaction_family":"addition"}</plan>', "plan")
        self.assertEqual(plan["reaction_family"], "addition")
        bare_plan = parse_json_tag(
            '```json\n{"reaction_family":"substitution"}\n```', "plan"
        )
        self.assertEqual(bare_plan["reaction_family"], "substitution")
        proposal = parse_executor_output(
            "<event_summary>attack</event_summary><plan_objection></plan_objection><action>ReactionTransformation(...)</action>",
            proposal_id="p1",
            directive_id="d1",
            review_round=1,
        )
        self.assertEqual(proposal.event_summary, "attack")
        self.assertEqual(proposal.action_pgl, "ReactionTransformation(...)")
        with self.assertRaises(ValueError):
            parse_json_tag("<plan>{}</plan><plan>{}</plan>", "plan")

    def test_openai_reasoning_is_separate_and_thinking_kwargs_are_sent(self):
        client = _Client()
        model = OpenAIChatModel(OpenAIChatConfig(), client=client)
        result = model.complete([{"role": "user", "content": "x"}], phase="test")
        self.assertEqual(result.reasoning, "internal reasoning")
        self.assertTrue(result.answer.startswith("<plan>"))
        self.assertEqual(
            client.chat.completions.kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )
        self.assertEqual(client.chat.completions.kwargs["max_tokens"], 8192)
        self.assertNotIn("messages", model.events[-1])
        self.assertEqual(model.events[-1]["answer"], result.answer)

    def test_planner_retries_malformed_protocol_without_thinking(self):
        model = _SequenceModel(
            [
                "thought without a final protocol block",
                """<plan>{
                    "reaction_family": "test",
                    "reaction_summary": "summary",
                    "global_mechanism": "mechanism",
                    "elementary_steps": [{"step_id": "E1", "description": "step"}],
                    "initial_elementary_step_id": "E1"
                }</plan>""",
            ]
        )
        plan = PlannerAgent(model).create_plan(self.task)
        self.assertEqual(plan.reaction_family, "test")
        self.assertEqual(model.calls, [None, False])
        self.assertIn("protocol_error", model.events[0])

    def test_directive_and_review_schema_guards(self):
        directive = StepDirective.from_dict(
            {
                "directive_decision": "EXECUTE",
                "active_elementary_step": {"step_id": "E1", "description": "Activate."},
                "current_position": "initial state",
                "expected_electron_transfer_event": "activate the electrophile",
            },
            plan=self.plan,
            mechanism_step_idx=0,
            directive_id="d2",
            expected_elementary_step=self.plan.elementary_step("E1"),
        )
        self.assertEqual(directive.active_elementary_step["step_id"], "E1")
        replan_directive = StepDirective.from_dict(
            {
                "directive_decision": "REPLAN",
                "active_elementary_step": {"step_id": "E1", "description": "Activate."},
                "current_position": "step is incompatible",
                "expected_electron_transfer_event": "revise the remainder",
            },
            plan=self.plan,
            mechanism_step_idx=1,
            directive_id="d3",
            expected_elementary_step=self.plan.elementary_step("E1"),
        )
        self.assertEqual(replan_directive.directive_decision, "REPLAN")
        verbose_execute = StepDirective.from_dict(
            {
                "directive_decision": "EXECUTE, because the step is applicable",
                "active_elementary_step": {"step_id": "E1", "description": "Activate."},
                "current_position": "initial state",
                "expected_electron_transfer_event": "activate the electrophile",
            },
            plan=self.plan,
            mechanism_step_idx=0,
            directive_id="d4",
            expected_elementary_step=self.plan.elementary_step("E1"),
        )
        self.assertEqual(verbose_execute.directive_decision, "EXECUTE")
        review = PlannerReview.from_dict(
            {
                "decision": "APPROVE",
                "global_consistency": True,
                "elementary_step_consistency": True,
                "electron_event_consistency": True,
                "chemical_assessment": "consistent",
                "plan_status": "KEEP",
            },
            proposal_id="p2",
            review_id="r2",
        )
        self.assertEqual(review.decision, "APPROVE")
        with self.assertRaises(ValueError):
            PlannerReview.from_dict(
                {
                    "decision": "APPROVE",
                    "global_consistency": False,
                    "elementary_step_consistency": True,
                    "electron_event_consistency": True,
                    "chemical_assessment": "inconsistent",
                },
                proposal_id="p3",
                review_id="r3",
            )

    def test_first_version_prompts_have_pgl_guides_and_no_skill_payloads(self):
        planner_prompt = "\n".join(item["content"] for item in create_plan_messages(self.task))
        executor_prompt = "\n".join(
            item["content"]
            for item in executor_messages(
                self.task,
                self.plan,
                self.directive,
                "ReactionState(current)",
                [],
                [],
                None,
            )
        )
        self.assertIn("Substructure.atomdict", planner_prompt)
        self.assertIn("ElectronTransfer", executor_prompt)
        self.assertNotIn("available_plan_skills", planner_prompt)
        self.assertNotIn("retrieved_action_skills", executor_prompt)
        self.assertNotIn("derived_target_change_summary", planner_prompt)
        self.assertIn("Each elementary_steps item must describe exactly one", planner_prompt)
        self.assertIn("explicitly perform an internal coverage audit", planner_prompt)
        self.assertIn("catalyst or reagent regeneration", planner_prompt)
        self.assertIn("they are not required to enumerate fixed atom labels", planner_prompt)
        self.assertIn("possible dataset-scope or annotation boundary", planner_prompt)
        self.assertIn("Hcount is normally absent when its stored value is zero", planner_prompt)

    def test_legacy_macro_plan_is_readable_but_serializes_new_protocol(self):
        plan = GlobalPlan.from_dict(
            {
                "reaction_family": "legacy",
                "reaction_summary": "summary",
                "global_mechanism": "mechanism",
                "macro_steps": [{"step_id": "M1", "description": "legacy step"}],
                "initial_macro_step_id": "M1",
            },
            plan_version=1,
        )
        rendered = plan.to_dict()
        self.assertEqual(plan.elementary_steps[0]["step_id"], "M1")
        self.assertIn("elementary_steps", rendered)
        self.assertNotIn("macro_steps", rendered)

    def test_posthoc_gt_assessment_is_structured_and_separate(self):
        answer = """<gt_assessment>{
          "planner_global_mechanism_judgment": "correct",
          "planner_elementary_steps_judgment": "partially_correct",
          "predicted_events_judgment": "incorrect",
          "gt_mechanism_summary": "GT describes activation followed by substitution.",
          "planner_global_assessment": "The overall substitution mechanism is correct.",
          "planner_elementary_steps_assessment": "The activation event is missing.",
          "predicted_events_assessment": "The executed path omitted activation.",
          "concise_conclusion": "总体路线部分正确，但逐步事件缺失活化步骤。"
        }</gt_assessment>"""
        model = _SequenceModel([answer])
        result = PosthocGroundTruthEvaluator(model).assess(
            task=self.task,
            initial_plan=self.plan,
            predicted_actions_pgl=["ReactionTransformation(predicted)"],
            gold_actions_pgl=["ReactionTransformation(gold)"],
            rollout_outcome={"success": False},
        )
        self.assertIsInstance(result, GroundTruthAssessment)
        self.assertEqual(result.planner_global_mechanism_judgment, "correct")
        self.assertEqual(result.planner_mechanism_judgment, "correct")
        self.assertEqual(
            result.planner_elementary_steps_judgment, "partially_correct"
        )
        self.assertEqual(result.predicted_events_judgment, "incorrect")
        prompt = model.events[0]["phase"]
        self.assertEqual(prompt, "posthoc.gt_assessment")
        evaluator_system = model.message_batches[0][0]["content"]
        self.assertIn("planner_global_mechanism_judgment", evaluator_system)
        self.assertIn("planner_elementary_steps_judgment", evaluator_system)
        self.assertIn("after normalization nothing may be added", evaluator_system)
        self.assertIn("chemical-environment-equivalent", evaluator_system)


if __name__ == "__main__":
    unittest.main()
