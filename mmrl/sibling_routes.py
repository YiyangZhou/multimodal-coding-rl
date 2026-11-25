"""Sibling-level harness selection; integrate before Gym dispatch, never after scoring."""
from copy import deepcopy

ROUTES = ('single', 'opencode', 'codex', 'hermes')
SCHEDULE = tuple(route for route in ROUTES for _ in range(2))


def apply_variant(row, variants, *, siblings=8):
    """Change the agent request without changing group, rollout or task identity.

    Variants are prebuilt, audited Gym rows for the same canonical task. The
    caller stamps its durable sibling index before calling this function. A
    retry of only some siblings therefore selects the same original route.
    """
    if siblings != 8 or set(variants) != set(ROUTES):
        raise ValueError('Mixed groups require eight siblings and four routes')
    index = row.get('_rowidx')
    if type(index) is not int or not 0 <= index < siblings:
        raise ValueError('A stable sibling index in [0, 8) is required')
    for route, variant in variants.items():
        if variant.get('task') != row.get('task'):
            raise ValueError('All sibling variants must use the same complete task')
        if variant.get('route') != route or variant.get('agent_ref') != {
            'type': 'responses_api_agents', 'name': 'mm_' + route
        }:
            raise ValueError('Variant route and Gym agent disagree')
        if not variant.get('responses_create_params', {}).get('input'):
            raise ValueError('Every variant needs its own actual request')
    selected = SCHEDULE[index]
    variant = variants[selected]
    output = deepcopy(row)
    output['route'] = selected
    output['_mixed_sibling_route'] = selected
    output['agent_ref'] = deepcopy(variant['agent_ref'])
    # Preserve runtime sampling/capture fields; replace only route-specific
    # messages and respect the smaller of the prepared and runtime budgets.
    params = output['responses_create_params']
    params['input'] = deepcopy(variant['responses_create_params']['input'])
    params['max_output_tokens'] = min(params['max_output_tokens'],
                                      variant['responses_create_params']['max_output_tokens'])
    output.pop('task_source', None)  # Explicit agent_ref is authoritative.
    output.pop('sibling_variants', None)
    return output


def validate_resolved_agent(row, resolved_agent_ref):
    """Fail before a streamed result can be persisted under the wrong harness."""
    route = row.get('_mixed_sibling_route')
    if route is None:
        return
    if route not in ROUTES or row.get('route') != route:
        raise ValueError('Mixed sibling route metadata disagrees')
    expected = {'type': 'responses_api_agents', 'name': 'mm_' + route}
    if row.get('agent_ref') != expected or resolved_agent_ref != expected:
        raise ValueError('Gym returned a different agent for a mixed sibling')


def mixed_metrics(completions, inputs, compute, namespace):
    """Keep group-wide metrics global and attribute agent metrics per sibling."""
    ordered = sorted(inputs, key=lambda row: row['_rowidx'])
    if not completions or len(completions) != len(ordered):
        raise ValueError('Mixed metric rows do not match completions')
    if len({r['_rowidx'] for r in ordered}) != len(ordered):
        raise ValueError('Mixed metric sibling indices are duplicated')
    grouped = {}
    for row, completion in zip(ordered, completions):
        route = row.get('_mixed_sibling_route')
        if route not in ROUTES or row.get('route') != route:
            raise ValueError('Mixed metric row has an invalid route')
        grouped.setdefault(namespace(row), []).append(completion)
    metrics = compute(completions, 'mixed')
    for agent, subset in grouped.items():
        local = compute(subset, agent)
        metrics.update({k:v for k,v in local.items() if k.startswith(agent + '/')})
        metrics[agent + '/sibling_share'] = len(subset) / len(completions)
    return metrics
