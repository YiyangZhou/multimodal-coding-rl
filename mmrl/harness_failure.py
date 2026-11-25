"""Translate a harness process failure into a verifier-visible failed response.

Only the upstream harness runner's explicit failure is handled. Transport,
configuration and unrelated runtime errors retain their original behavior.
"""

from time import time
from uuid import uuid4


def runner_failure_response(error, params, response_type):
    """Return a credential-free failed response, or None for unrelated errors."""
    if type(error) is not RuntimeError or not str(error).startswith('runner failed ('):
        return None
    return response_type(
        id='resp_' + uuid4().hex,
        created_at=time(),
        model=params.model or 'policy',
        object='response',
        status='failed',
        error={'code': 'server_error', 'message': 'harness_runner_failed'},
        metadata={'failed': 'true'},
        output=[],
        tool_choice=params.tool_choice,
        tools=params.tools,
        parallel_tool_calls=params.parallel_tool_calls,
    )
