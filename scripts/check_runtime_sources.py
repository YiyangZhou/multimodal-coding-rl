"""Check native patch prerequisites without changing either upstream checkout."""
import argparse
import hashlib
import json
from pathlib import Path


def check_profile(root, manifest):
    files = []
    for relative, expected in manifest['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Patch manifest contains an invalid relative path')
        target = root / path
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        # A null before hash means the patch intentionally adds a new file.
        state = ('applied' if actual == expected['after_sha256'] else
                 'ready' if actual == expected['before_sha256'] else
                 'missing' if actual is None else 'different')
        files.append({'path': relative, 'state': state})
    states = {item['state'] for item in files}
    status = 'applied' if states == {'applied'} else 'ready' if states == {'ready'} else 'incompatible'
    return {'status': status, 'files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl-root', required=True, type=Path)
    parser.add_argument('--gym-root', type=Path)
    parser.add_argument('--gym-manifest', type=Path, help='External Gym verification manifest (not distributed here)')
    parser.add_argument('--public-runtime', action='store_true', help='Inspect the experimental composite public-source profile')
    args = parser.parse_args()
    patches = Path(__file__).resolve().parents[1] / 'patches'
    results = {}
    profiles = [('rl', args.rl_root, patches / ('public-native-runtime.json' if args.public_runtime else 'automodel-captured-images.json'))]
    if args.gym_root:
        if not args.gym_manifest:
            parser.error('--gym-root requires --gym-manifest from the external runtime')
        profiles.append(('gym', args.gym_root, args.gym_manifest))
    for label, root, manifest_path in profiles:
        if not root.is_dir():
            results[label] = {'status': 'missing_checkout', 'files': []}
        elif not manifest_path.is_file():
            results[label] = {'status': 'missing_manifest', 'files': []}
        else:
            results[label] = check_profile(root, json.loads(manifest_path.read_text()))
    print(json.dumps(results, indent=2))
    return 0 if all(x['status'] in {'ready', 'applied'} for x in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
