"""Verify pinned interfaces and explicitly selected native integration layers."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def manifest(directory, name):
    result = json.loads((directory / f'{name}.json').read_text())
    if digest(directory / f'{name}.patch') != result['patch_sha256']:
        raise SystemExit(f'{name} patch does not match its manifest')
    return result['files']


def check_sources(root, pinned, base_layer, optional_layers):
    expected = dict(pinned)
    # The source lock already records applied capture/protocol files where
    # those overlap. Refuse inconsistent pins instead of silently overriding.
    for relative, spec in base_layer.items():
        if relative in expected and expected[relative] != spec['after_sha256']:
            raise SystemExit('Base integration disagrees with source lock: ' + relative)
        expected[relative] = spec['after_sha256']
    for layer in optional_layers:
        for relative, spec in layer.items():
            if relative in expected and expected[relative] != spec['before_sha256']:
                raise SystemExit('Optional patch does not follow its predecessor: ' + relative)
            expected[relative] = spec['after_sha256']
    for relative, wanted in expected.items():
        if digest(root / relative) != wanted:
            raise SystemExit(f'Source mismatch: {root.name}/{relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl', type=Path, required=True)
    parser.add_argument('--gym', type=Path)
    parser.add_argument('--gym-source-lock', type=Path, help='External path-to-SHA256 JSON mapping')
    parser.add_argument('--linear-logprobs', action='store_true')
    parser.add_argument('--process-returns', action='store_true')
    parser.add_argument('--process-tools', action='store_true')
    parser.add_argument('--local-padding', action='store_true')
    parser.add_argument('--media-custody', action='store_true')
    args = parser.parse_args()
    if args.media_custody and not args.process_returns:
        parser.error('--media-custody follows the process-returns integration layer')
    if args.local_padding and not args.linear_logprobs:
        parser.error('--local-padding requires --linear-logprobs')
    root = Path(__file__).resolve().parents[1]
    directory = root / 'patches'
    lock = json.loads((root / 'source.lock.json').read_text())
    rl_layers = []
    if args.linear_logprobs:
        rl_layers.append(manifest(directory, 'automodel-linear-logprobs'))
    if args.local_padding:
        rl_layers.append(manifest(directory, 'automodel-local-padding'))
    if args.process_returns:
        rl_layers.append(manifest(directory, 'native-process-returns'))
    if args.media_custody:
        rl_layers.append(manifest(directory, 'native-media-custody'))
    if args.process_tools and not args.gym:
        parser.error('--process-tools requires --gym')
    gym_directory = Path(os.environ.get('MMRL_EXTERNAL_GYM_PATCH_DIR', ''))
    if args.gym and (not args.gym_source_lock or not os.environ.get('MMRL_EXTERNAL_GYM_PATCH_DIR')):
        parser.error('--gym requires --gym-source-lock and MMRL_EXTERNAL_GYM_PATCH_DIR')
    gym_layers = [manifest(gym_directory, 'gym-process-tools')] if args.process_tools else []
    check_sources(args.rl, lock['sha256']['nemo_rl'],
                  manifest(directory, 'automodel-captured-images'), rl_layers)
    if args.gym:
        from prepare_image import load_source_pins
        check_sources(args.gym, load_source_pins(args.gym_source_lock),
                      manifest(gym_directory, 'gym-harness-protocol'), gym_layers)
    print('Pinned interfaces, dependency locks and selected native integration layers match')


if __name__ == '__main__':
    main()
