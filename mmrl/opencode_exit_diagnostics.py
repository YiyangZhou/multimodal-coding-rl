"""Bounded, payload-free diagnostics. No exception text crosses this boundary."""
import json
import sqlite3
from pathlib import Path

FILENAME = 'opencode-exit-diagnostics.json'
ERRORS = ('APIError', 'ContextOverflowError', 'MessageAbortedError', 'UnknownError', 'ProviderAuthError', 'OutputLengthError')
FINISHES = ('stop', 'tool-calls', 'length', 'content-filter', 'error', 'unknown')
FLAGS = ('permission', 'denied', 'context', 'timeout', 'rate_limit')

def validate(value):
    keys = {'schema_version', 'exit_code', 'timeout', 'root_error', 'http_status', 'last_finish', 'stderr_flags', 'collection'}
    if not isinstance(value, dict) or set(value) != keys:
        return None
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        return None
    if value['exit_code'] is not None and (type(value['exit_code']) is not int or not -255 <= value['exit_code'] <= 255):
        return None
    if type(value['timeout']) is not bool:
        return None
    if value['root_error'] not in (None, 'other', *ERRORS) or value['last_finish'] not in (None, 'other', *FINISHES):
        return None
    if value['http_status'] is not None and (type(value['http_status']) is not int or not 100 <= value['http_status'] <= 599):
        return None
    if value['collection'] not in ('ok', 'db_missing', 'db_error'):
        return None
    flags = value['stderr_flags']
    if not isinstance(flags, dict) or set(flags) != set(FLAGS) or any(type(v) is not bool for v in flags.values()):
        return None
    return value

def collect(db_path, exit_code, timed_out, stderr):
    # Read at most 64 KiB of already-buffered stderr, emit only fixed booleans.
    lowered = stderr[:65536].lower() if isinstance(stderr, bytes) else b''
    result = dict(schema_version=1, exit_code=exit_code if type(exit_code) is int and -255 <= exit_code <= 255 else None,
                  timeout=bool(timed_out), root_error=None, http_status=None, last_finish=None,
                  stderr_flags={k: k.encode() in lowered for k in FLAGS}, collection='ok')
    if not Path(db_path).is_file():
        result['collection'] = 'db_missing'
        return result
    con = None
    try:
        con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        for (raw,) in con.execute('select m.data from message m join session s on m.session_id = s.id where s.parent_id is null order by m.time_created'):
            msg = json.loads(raw)
            if not isinstance(msg, dict) or msg.get('role') != 'assistant':
                continue
            finish = msg.get('finish')
            result['last_finish'] = finish if finish in FINISHES else None if finish is None else 'other'
            error = msg.get('error')
            if isinstance(error, dict) and result['root_error'] is None:
                name = error.get('name')
                result['root_error'] = name if name in ERRORS else 'other'
                data = error.get('data')
                status = data.get('statusCode') if isinstance(data, dict) else None
                result['http_status'] = status if type(status) is int and 100 <= status <= 599 else None
    except Exception:
        result['collection'] = 'db_error'
    finally:
        if con is not None:
            con.close()
    return result

def save(path, db_path, exit_code, timed_out, stderr):
    try:
        value = collect(db_path, exit_code, timed_out, stderr)
        Path(path).write_text(json.dumps(value, separators=(',', ':')) + '\n')
    except Exception:
        # Diagnostic I/O must never replace the original outcome/exception.
        pass
