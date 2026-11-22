import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import prepare
import run


def test_reasoning_budget_supports_gateway_and_standard_usage():
    assert run.generation_tokens({'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 80,
                                  'completion_tokens_details': {'reasoning_tokens': 50}}) == 70
    assert run.generation_tokens({'prompt_tokens': 10, 'completion_tokens': 70, 'total_tokens': 80,
                                  'completion_tokens_details': {'reasoning_tokens': 50}}) == 70


def test_uniform_hydrogen_representation_ignores_selective_explicit_labels():
    c = prepare.chem
    def state(atoms, bonds):
        return c.ReactionState([c.Species('reactant', 'stable', c.Molecule('', [c.Substructure('', atoms, bonds)]))])
    implicit = state({'C1': c.Atom('C'), 'O2': c.Atom('O')}, [c.Bond('C1', 'O2', 'single')])
    selective = state({'C1': c.Atom('C'), 'O2': c.Atom('O'), 'H900001': c.Atom('H')},
                      [c.Bond('C1', 'O2', 'single'), c.Bond('O2', 'H900001', 'single')])
    a, _, _ = prepare.normalize_endpoints(implicit, implicit, 1)
    b, _, _ = prepare.normalize_endpoints(selective, selective, 1)
    assert repr(a) == repr(b)
    assert sum(atom.element == 'H' for _, atom in c.pgl.iter_atoms(a)) == 4
    assert all(not atom.Hcount for _, atom in c.pgl.iter_atoms(a))


def test_direct_path_parses_only_actions_not_arbitrary_code():
    valid = {'actions': [{'transformation_type': 'reaction', 'electron_transfers': [
        {'source': ['O1', 'lone_pair'], 'target': ['O1', 'C2', 'new_sigma_bond'],
         'electrons': 2, 'kind': 'bond_forming'}]}]}
    assert len(run.parse_path('<path>' + json.dumps(valid) + '</path>', prepare.chem)) == 1
    import pytest
    with pytest.raises(ValueError):
        run.parse_path(json.dumps({'actions': ["__import__('os').system('false')"]}), prepare.chem)


def test_frozen_inputs_are_disjoint_from_prior_ids():
    out = prepare.ROOT / 'outputs/publication_exp1_20261018'
    manifest = json.loads((out / 'manifest.json').read_text())
    gold = [json.loads(x) for x in (out / 'gold.jsonl').read_text().splitlines()]
    assert len(gold) == 80
    assert len({r['source_reaction_id'] for r in gold}) == 80
    assert not {r['source_reaction_id'] for r in gold}.intersection(manifest['excluded_prior_ids'])
    inputs = [json.loads(x) for x in (out / 'inputs.jsonl').read_text().splitlines()]
    assert all(set(r) == {'case_id', 'initial_state_pgl', 'target_state_pgl'} for r in inputs)


def test_scoring_restores_blinded_heavy_labels_but_anonymizes_hydrogen():
    import summarize
    c = prepare.chem
    from develop.metrics import compare_path
    predicted = c.ReactionTransformation('reaction', [c.ElectronTransfer(
        source=('O3', 'lone_pair'), target=('O3', 'H2', 'new_sigma_bond'), electrons=2, kind='bond_forming')])
    restored = summarize.translate(predicted, {'O3': 'O205'})
    gold = c.ReactionTransformation('reaction', [c.ElectronTransfer(
        source='O205', target='H900003', electrons=2, kind='bond_forming')])
    assert compare_path([restored], [gold]).step_partition_match
    assert predicted.electron_transfers[0].source[0] == 'O3'


def test_direct_self_review_finishes_before_any_environment_replay():
    import importlib
    from types import SimpleNamespace
    out = prepare.ROOT / 'outputs/publication_exp1_20261018'
    sys.path.insert(0, str(out / 'snapshot'))
    bridge = importlib.import_module('exp1_agent.chemistry').ChemistryBridge(str(out / 'snapshot'))
    local = sys.modules['standalone_chemistry']
    def state(ccharge, ocharge, connected):
        sub = local.Substructure('', {'C1': local.Atom('C', charge=ccharge),
                                       'O2': local.Atom('O', charge=ocharge)},
                                 [local.Bond('C1', 'O2', 'single')] if connected else [])
        return repr(local.ReactionState([local.Species('reactant', 'stable', local.Molecule('', [sub]))]))
    task = SimpleNamespace(task_description='Infer mechanism', initial_state_pgl=state(1, -1, False),
                           target_state_pgl=state(0, 0, True))
    data = {'actions': [{'electron_transfers': [{'source': ['O2', 'lone_pair'],
                'target': ['O2', 'C1', 'new_sigma_bond'], 'electrons': 2, 'kind': 'bond_forming'}]}]}
    class Model:
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(answer=json.dumps(data))
    model = Model()
    class GuardedBridge:
        def __getattr__(self, name):
            assert model.calls == 4, 'simulator accessed before all direct self-reviews completed'
            return getattr(bridge, name)
    modules = {'local': local, 'prompts': importlib.import_module('exp1_agent.prompts'),
               'skills': importlib.import_module('exp1_agent.planner_skills')}
    result = run.run_direct(task, model, modules, GuardedBridge())
    assert result['runtime_success']
    assert len(result['accepted_actions_pgl']) == 1


def test_progress_reader_tolerates_resume_archiving_a_pending_result(tmp_path):
    from unittest.mock import patch
    import supervise
    p = tmp_path / 'direct' / 'exp1_001_r1.json'
    p.parent.mkdir()
    p.write_text(json.dumps({'finished': False}))
    with patch.object(Path, 'read_text', side_effect=FileNotFoundError('archived during resume')):
        assert supervise.counts(tmp_path)['direct'] == 0


def test_report_error_does_not_abort_background_inference(tmp_path):
    from unittest.mock import patch
    from types import SimpleNamespace
    import supervise
    with patch.object(supervise.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
        assert not supervise.refresh_report(HERE, tmp_path)
