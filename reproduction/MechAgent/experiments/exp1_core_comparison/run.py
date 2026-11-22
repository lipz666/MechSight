"""Run the frozen three-arm comparison; inference never opens gold.jsonl."""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


class BudgetExceeded(RuntimeError):
    pass


INFRA_ERRORS = ('APIConnectionError', 'APITimeoutError', 'RateLimitError',
                'InternalServerError', 'AuthenticationError', 'PermissionDeniedError',
                'usage is missing')


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    rendered = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    secret = os.environ.get('OPENAI_API_KEY')
    if secret:
        rendered = rendered.replace(secret, '[REDACTED]')
    tmp.write_text(rendered)
    tmp.replace(path)


def generation_tokens(usage):
    # This gateway reports reasoning separately from completion_tokens.
    # Other providers include it; total-prompt reconciles both conventions.
    completion = int(usage.get('completion_tokens') or 0)
    total = int(usage.get('total_tokens') or 0)
    prompt = int(usage.get('prompt_tokens') or 0)
    details = usage.get('completion_tokens_details') or {}
    reasoning = int(details.get('reasoning_tokens') or 0)
    return max(completion, total - prompt, reasoning)


class BudgetModel:
    def __init__(self, args, trace):
        from openai import OpenAI
        self.args = args
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=args.base_url,
                             timeout=args.timeout, max_retries=0)
        self.events = []
        self.trace = trace
        self.used = 0
        self.attempts = 0
        self.api_failures = 0

    def complete(self, messages, *, phase, enable_thinking=None):
        remaining = self.args.generation_budget - self.used
        if remaining < 256 or self.attempts >= self.args.max_calls:
            raise BudgetExceeded('common rollout generation-token/call limit reached')
        limit = min(self.args.direct_max_tokens if phase.startswith('direct.') else self.args.max_tokens, remaining)
        self.attempts += 1
        start = time.monotonic()
        response = None
        for attempt in range(12):
            try:
                stream = self.client.chat.completions.create(
                    model=self.args.model, messages=messages, max_tokens=limit,
                    temperature=0.2, reasoning_effort='high', stream=True,
                    stream_options={'include_usage': True})
                content, reasoning, usage, returned_model, finish = [], [], None, None, None
                for chunk in stream:
                    returned_model = chunk.model
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta.model_dump()
                    if delta.get('content'):
                        content.append(delta['content'])
                    if delta.get('reasoning_content') or delta.get('reasoning'):
                        reasoning.append(delta.get('reasoning_content') or delta['reasoning'])
                    finish = choice.finish_reason or finish
                response = SimpleNamespace(model=returned_model, usage=SimpleNamespace(model_dump=lambda: usage),
                    choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(model_dump=lambda:
                        {'content': ''.join(content), 'reasoning_content': ''.join(reasoning)}))])
                break
            except Exception as exc:
                self.api_failures += 1
                status = getattr(exc, 'status_code', None)
                event = {'phase': phase, 'api_error': type(exc).__name__, 'status': status,
                         'retry': attempt, 'detail': str(exc).replace(os.environ['OPENAI_API_KEY'], '[REDACTED]')[:800],
                         'elapsed_seconds': time.monotonic() - start}
                self._save(event)
                if attempt == 11 or (status and status not in {408, 429, 500, 502, 503, 504}):
                    raise
                time.sleep(min(30, 5 * 2 ** attempt))
        choice = response.choices[0]
        message = choice.message.model_dump()
        answer = message.get('content') or ''
        if isinstance(answer, list):
            answer = '\n'.join(x.get('text', '') for x in answer if isinstance(x, dict))
        usage = (response.usage.model_dump() if response.usage else {}) or {}
        if not usage:
            raise RuntimeError('usage is missing; equal token budget cannot be audited')
        consumed = generation_tokens(usage)
        self.used += consumed
        event = {'phase': phase, 'requested_model': self.args.model, 'returned_model': response.model,
                 'reasoning_effort': 'high', 'finish_reason': choice.finish_reason,
                 'request_max_tokens': limit, 'usage': usage, 'generation_tokens': consumed,
                 'elapsed_seconds': time.monotonic() - start,
                 'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                 'answer': answer, 'reasoning': message.get('reasoning_content') or message.get('reasoning')}
        self.events.append(event)
        self._save(event)
        if not str(answer).strip():
            raise ValueError('empty_content')
        return SimpleNamespace(answer=answer, reasoning=event['reasoning'], usage=usage,
                               finish_reason=choice.finish_reason, message=message)

    def _save(self, event):
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        with self.trace.open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')


def parse_path(answer, local):
    value = answer.strip()
    if '<path>' in value:
        value = value.split('<path>', 1)[1].split('</path>', 1)[0].strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    data = json.loads(value)
    actions = data.get('actions')
    if not isinstance(actions, list) or not actions or len(actions) > 12:
        raise ValueError('actions must be a non-empty list of at most 12 elementary steps')
    result = []
    for action in actions:
        if isinstance(action, str):
            parsed = local.parse_pgl_literal(action)
            if not isinstance(parsed, local.ReactionTransformation):
                raise ValueError('expected ReactionTransformation')
        elif isinstance(action, dict):
            transfers = []
            for t in action['electron_transfers']:
                transfers.append(local.ElectronTransfer(source=t['source'], target=t['target'],
                                                        electrons=t['electrons'], kind=t['kind']))
            parsed = local.ReactionTransformation(action.get('transformation_type', 'reaction'), transfers)
        else:
            raise ValueError('invalid action type')
        result.append(repr(parsed))
    return result


def run_direct(task, model, modules, bridge):
    prompts, local = modules['prompts'], modules['local']
    schema = '''Return <path>{"mechanism_summary":"...","actions":[{"transformation_type":"reaction","electron_transfers":[{"source":["O1","lone_pair"],"target":["O1","C2","new_sigma_bond"],"electrons":2,"kind":"bond_forming"}]}]}</path>.
Use actual provided atom labels. Arrays in endpoints encode PGL structural tuples. Return the COMPLETE path, not just one step. At most 12 elementary steps. Include all necessary coupled arrows. No GT path or step count is available.'''
    system = '\n\n'.join([prompts.PGL_STATE_GUIDE, prompts.PGL_ACTION_GUIDE,
                           prompts.PLANNER_DECOMPOSITION_GUIDE, schema])
    # Identical endpoint-derived evidence is available to the direct baseline.
    facts = modules['skills'].pgl_structure_change_facts(task.initial_state_pgl, task.target_state_pgl)
    user = (task.task_description + '\nINITIAL:\n' + task.initial_state_pgl +
            '\nTARGET:\n' + task.target_state_pgl + '\nEndpoint-derived facts:\n' + json.dumps(facts))
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    versions, errors = [], []
    # Four full-path proposals, with self-review but no simulator/validator feedback.
    for turn in range(4):
        try:
            response = model.complete(messages, phase=f'direct.path_{turn + 1}')
        except BudgetExceeded as exc:
            errors.append({'phase': 'budget', 'error': str(exc)})
            break
        parsed = None
        try:
            parsed = parse_path(response.answer, local)
            versions.append({'turn': turn + 1, 'actions_pgl': parsed})
        except Exception as exc:
            errors.append({'phase': f'parse_{turn + 1}', 'error': str(exc)})
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': response.answer},
                    {'role': 'user', 'content': 'Independently audit your entire path for mechanistic causality, omitted events, exact atom labels, electron sources, proton conservation and step boundaries. Correct errors if needed. Return the complete final path in the required schema. You have no environment execution feedback.'}]
    proposed = versions[-1]['actions_pgl'] if versions else []
    # Replay is performed only AFTER all direct-model calls; it cannot inform self-review.
    replay = replay_actions(task, proposed, bridge, local)
    return {'accepted_actions_pgl': replay['accepted'], 'proposed_actions_pgl': proposed,
            'path_versions': versions, 'validation_errors': replay['errors'], 'errors': errors,
            'stop_reason': replay['stop_reason'], 'final_state_pgl': replay['final_state_pgl'],
            'runtime_success': replay['strict_target_match'], 'planner_plan': {}}


def replay_actions(task, actions, bridge, local):
    envtask = SimpleNamespace(path=SimpleNamespace(sequence=[local.parse_pgl_literal(task.initial_state_pgl),
                                                            local.parse_pgl_literal(task.target_state_pgl)]))
    env = bridge.make_env(envtask, max_env_calls=12)
    env.reset()
    accepted, errors = [], []
    stop = 'complete_path_replayed'
    for index, text in enumerate(actions):
        action, parsed = bridge.parse_action(text)
        valid = bridge.validate_action(action, env.current_state, env.legal_local_actions()) if parsed['valid'] else parsed
        if not valid.get('valid'):
            errors.append({'step': index + 1, 'validation': valid})
            stop = 'invalid_action'
            break
        preview = env.preview_action(action)
        if not preview.get('preview_valid'):
            errors.append({'step': index + 1, 'preview': preview})
            stop = 'invalid_edit'
            break
        _, _, _, info = env.step(action)
        if not info.get('edit_valid'):
            errors.append({'step': index + 1, 'execution': info})
            stop = 'invalid_edit'
            break
        accepted.append(text)
    return {'accepted': accepted, 'errors': errors, 'stop_reason': stop,
            'final_state_pgl': bridge.state_to_pgl(env.current_state), 'strict_target_match': env.product_reached()}


def run_job(job, args, modules, bridge):
    arm, row, repeat = job
    stem = f"{row['case_id'].replace('::', '_')}_r{repeat}"
    destination = args.output / arm / (stem + '.json')
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get('finished'):
            return arm, row['case_id'], repeat, 'cached'
        archived = destination.parent / 'infrastructure_attempts' / (stem + f'_{int(time.time())}.json')
        archived.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(archived)
    start = time.monotonic()
    trace = args.output / arm / 'traces' / (stem + '.jsonl')
    if trace.exists():
        trace.rename(trace.with_name(stem + f'.previous_{int(time.time())}.jsonl'))
    model = BudgetModel(args, trace)
    task = modules['schemas'].MechanismTaskContext(
        reaction_id=row['case_id'], task_description='Using only the initial and target molecular graphs, infer a complete stepwise elementary electron-transfer mechanism. Reaction name, class, conditions, GT intermediates and GT step count are hidden.',
        initial_state_pgl=row['initial_state_pgl'], target_state_pgl=row['target_state_pgl'],
        reaction_metadata={'input_mode': 'structures_only'})
    error = None
    try:
        if arm == 'direct':
            result = run_direct(task, model, modules, bridge)
        else:
            library = modules['skills'].PlannerSkillLibrary.from_directory(args.output / 'snapshot/exp1_agent/planner_skills') if arm == 'agent_skills79' else None
            planner = modules['agents'].PlannerAgent(model, skill_library=library)
            executor = modules['agents'].ExecutorAgent(model)
            envtask = SimpleNamespace(path=SimpleNamespace(sequence=[modules['local'].parse_pgl_literal(task.initial_state_pgl),
                                                                    modules['local'].parse_pgl_literal(task.target_state_pgl)]))
            env = bridge.make_env(envtask, max_env_calls=12)
            runtime = modules['orchestrator'].PlannerExecutorAgent(
                env=env, planner=planner, executor=executor, state_to_pgl=bridge.state_to_pgl,
                parse_action=bridge.parse_action, validate_action=bridge.validate_action,
                config=modules['orchestrator'].PlannerExecutorConfig())
            run = runtime.run(task)
            result = {'accepted_actions_pgl': run.state.environment.history_actions_pgl,
                      'proposed_actions_pgl': [p['action_pgl'] for p in run.executor_proposals if p.get('action_pgl')],
                      'stop_reason': run.state.control.stop_reason, 'runtime_success': run.state.control.success,
                      'final_state_pgl': bridge.state_to_pgl(env.current_state),
                      'planner_plan': run.plan_versions[0] if run.plan_versions else {},
                      'errors': run.errors, 'trajectory': run.to_dict()}
    except Exception as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc)[:600]}
        result = {'accepted_actions_pgl': [], 'proposed_actions_pgl': [], 'stop_reason': 'infrastructure_error',
                  'runtime_success': False, 'final_state_pgl': row['initial_state_pgl'], 'errors': [error]}
    result.update({'case_id': row['case_id'], 'arm': arm, 'repeat_index': repeat,
                   'finished': True, 'elapsed_seconds': time.monotonic() - start,
                   'generation_tokens': model.used, 'prompt_tokens': sum(int(e['usage'].get('prompt_tokens') or 0) for e in model.events),
                   'total_tokens': sum(int(e['usage'].get('total_tokens') or 0) for e in model.events),
                   'reasoning_tokens': sum(int((e['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens') or 0) for e in model.events),
                   'calls': len(model.events), 'api_failures': model.api_failures,
                   'requested_model': args.model, 'returned_models': sorted(set(e['returned_model'] for e in model.events)),
                   'generation_budget': args.generation_budget, 'error': error})
    if any(marker in json.dumps(result.get('errors', [])) for marker in INFRA_ERRORS):
        result['finished'] = False
        result['infrastructure_pending'] = True
    atomic_json(destination, result)
    return arm, row['case_id'], repeat, 'infrastructure_pending' if not result['finished'] else result['stop_reason']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--base-url', default='https://LLM-GATEWAY-2.example/v1')
    ap.add_argument('--model', default='gemini-3.8-flash-high')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--direct-max-tokens', type=int, default=8192)
    ap.add_argument('--generation-budget', type=int, default=98304)
    ap.add_argument('--max-calls', type=int, default=64)
    ap.add_argument('--limit-cases', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--arms', nargs='+', default=['direct', 'agent_no_skills', 'agent_skills79'])
    args = ap.parse_args()
    args.output = args.output.resolve()
    manifest = json.loads((args.output / 'manifest.json').read_text())
    for rel, expected in manifest['snapshot_sha256'].items():
        if hashlib.sha256((args.output / rel).read_bytes()).hexdigest() != expected:
            raise SystemExit('Frozen snapshot hash mismatch: ' + rel)
    if hashlib.sha256((args.output / 'inputs.jsonl').read_bytes()).hexdigest() != manifest['inputs_sha256']:
        raise SystemExit('Input hash mismatch')
    sys.path.insert(0, str(args.output / 'snapshot'))
    modules = {name: importlib.import_module('exp1_agent.' + name) for name in
               ['agents', 'schemas', 'orchestrator', 'prompts', 'planner_skills', 'chemistry']}
    modules['skills'] = modules['planner_skills']
    bridge = modules['chemistry'].ChemistryBridge(str(args.output / 'snapshot'))
    modules['local'] = sys.modules['standalone_chemistry']
    rows = [json.loads(line) for line in (args.output / 'inputs.jsonl').read_text().splitlines()]
    if args.limit_cases:
        rows = rows[:args.limit_cases]
    jobs = [(arm, row, repeat) for row in rows for repeat in range(1, args.repeats + 1) for arm in args.arms]
    random.Random(20261018).shuffle(jobs)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['notes'] = ['same 98304 observed generation-token stopping threshold including reasoning; no LLM GT evaluator',
                       'direct: four full-path proposals/self-reviews without simulator feedback',
                       'agent arms: original frozen planner/executor/review loop; max 12 executed steps',
                       'retrieval costs included in skills-arm budget; actual costs reported separately']
    configpath = args.output / ('smoke_config.json' if args.limit_cases else 'run_config.json')
    if configpath.exists() and json.loads(configpath.read_text()) != config:
        raise SystemExit('Refusing to resume with changed configuration.')
    atomic_json(configpath, config)
    atomic_json(args.output / 'STATUS.json', {'status': 'running', 'expected_rollouts': len(jobs), 'runner_stopped': False})
    (args.output / 'run.pid').write_text(str(os.getpid()))
    print(json.dumps({'event': 'started', 'jobs': len(jobs), 'workers': args.workers}), flush=True)
    done = 0
    consecutive_infra = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, j, args, modules, bridge) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            answer = future.result()
            done += 1
            consecutive_infra = consecutive_infra + 1 if answer[-1] == 'infrastructure_pending' else 0
            print(json.dumps({'event': 'completed', 'done': done, 'total': len(jobs), 'job': answer}), flush=True)
            atomic_json(args.output / 'progress.json', {'completed_this_invocation': done, 'jobs': len(jobs),
                                                       'updated_unix': time.time(), 'last_job': answer})
            if consecutive_infra >= 3:
                for pending in futures:
                    pending.cancel()
                print(json.dumps({'event': 'service_blocked', 'reason': 'three consecutive infrastructure failures; pending jobs retained for resume'}), flush=True)
                raise SystemExit(75)
    print(json.dumps({'event': 'finished', 'jobs': len(jobs)}), flush=True)


if __name__ == '__main__':
    main()
