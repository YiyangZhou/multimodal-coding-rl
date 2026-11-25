"""Export paired official charts after an explicit legacy-aware pixel overlap audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from PIL import Image
from mmrl.tasks import Task, gym_row
from mmrl.vision_data import checked_file

META_HASHES = {
    'dimentions_info.jsonl': '440b9749e4ce042a06b78b7ca5086ab32c322bc869b4878b4b47cd0a75c6031d',
    'dimentions_info_edit.jsonl': 'b346ba554dd0912d43a709a221e3e94e772caadf7ebe4386a7d20af09f6b68fd',
}


def prepare(archive_root, metadata_root, audit_path, output):
    audit = json.loads(audit_path.read_text())
    if (not audit.get('passed') or audit.get('rows') != 4800 or audit.get('paired_ids') != 2400
            or audit.get('split_kind') != 'custom_chart_group_split_with_legacy_pixel_constraints'):
        raise ValueError('Expected complete paired-chart overlap audit')
    inventory = json.loads((archive_root.parent / 'inventory.json').read_text())
    if inventory['archive_sha256'] != 'a9b774a165e2e534f02b269285736cb2eed6bdb433d603a66e835fd768336866':
        raise ValueError('Expected pinned official archive inventory')
    metadata = {}
    for category, name in zip(('direct', 'customized'), META_HASHES):
        raw = checked_file(metadata_root, name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != META_HASHES[name]:
            raise ValueError('Official prompt metadata changed')
        rows = [json.loads(line) for line in raw.splitlines()]
        metadata[category] = {row['idx']: row for row in rows}
        if len(rows) != 2400 or len(metadata[category]) != 2400:
            raise ValueError('Missing or duplicate official prompt identity')
    template_path = checked_file(metadata_root, 'chart2code/prompts/DirectAgent/gpt-4-vision-preview.json')
    template = json.loads(template_path.read_text())['instruction']
    if set(audit['assignments']) != set(metadata['direct']) or set(metadata['direct']) != set(metadata['customized']):
        raise ValueError('Overlap audit and official prompts disagree')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, excluded, groups = Counter(), [], {}
    files = {}
    with tempfile.TemporaryDirectory(prefix='.chart-prepare-', dir=output.parent) as temp:
        work = Path(temp)
        streams = {split: (work / f'{split}.gym.jsonl').open('wb') for split in ('train', 'validation')}
        digests = {split: hashlib.sha256() for split in streams}
        try:
            for identity, entry in sorted(audit['assignments'].items()):
                split = entry['split']
                if split == 'excluded_existing_train_validation_overlap':
                    excluded.append(identity)
                    continue
                if split not in streams:
                    raise ValueError('Unknown audited split')
                if groups.setdefault(entry['group'], split) != split:
                    raise ValueError('Paired/duplicate charts cross splits')
                images, codes = {}, {}
                for category in ('direct', 'customized'):
                    info = entry[category]
                    image = Path(info['image'])
                    code = Path(info['code'])
                    for path in (image, code):
                        relative = path.relative_to(archive_root).as_posix()
                        checked_file(archive_root, relative)
                    if image.stem != identity or code != image.with_suffix('.py'):
                        raise ValueError('Source mapping changed')
                    with Image.open(image) as im:
                        pixels = im.convert('RGB')
                        pixel_hash = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
                    if pixel_hash != info['pixels_sha256']:
                        raise ValueError('Audited image pixels changed')
                    images[category], codes[category] = str(image), code.read_text()
                    files[str(code)] = hashlib.sha256(code.read_bytes()).hexdigest()
                for category in ('direct', 'customized'):
                    row = metadata[category][identity]
                    prompt = (template.format(width=row['width'], height=row['height']) if category == 'direct'
                              else row['instruction'] + f"\nNote that it is necessary to use figsize=({row['width']}, {row['height']}) to set the image size.\n")
                    task = Task('chartmimic', category + '/' + identity, prompt,
                                [images['direct']], category=category, ground_truth=codes[category],
                                judge_references=[images[category]])
                    record = gym_row(task, 'single', 32768)
                    content = record['responses_create_params']['input'][-1]['content']
                    assert content[0] == {'type': 'input_text', 'text': prompt}
                    assert len(content) == 2 and content[1]['type'] == 'input_image'
                    payload = (json.dumps(record) + '\n').encode()
                    streams[split].write(payload)
                    digests[split].update(payload)
                    counts[split] += 1
        finally:
            for stream in streams.values():
                stream.close()
        if any(counts[split] != audit['counts'].get(split, 0) for split in streams):
            raise ValueError('Export count differs from overlap audit')
        report = {'benchmark': 'chartmimic', 'counts': dict(counts), 'excluded_ids': excluded,
                  'upstream_split': 'evaluation', 'split_kind': audit['split_kind'],
                  'source_revision': inventory['source_revision'], 'metadata_sha256': META_HASHES,
                  'template_sha256': hashlib.sha256(template_path.read_bytes()).hexdigest(),
                  'overlap_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                  'reference_code_sha256': files,
                  'gym_sha256': {k: v.hexdigest() for k, v in digests.items()},
                  'route': 'single', 'max_output_tokens': 32768, 'training_ready': False,
                  'note': 'Original image is policy input for both variants; customized target and code stay private. Exact pixel overlap only; native admission pending.'}
        (work / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
        work.rename(output)
    return {k: v for k, v in report.items() if k != 'reference_code_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--metadata-root', type=Path, required=True)
    parser.add_argument('--overlap-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive_root.resolve(), args.metadata_root.resolve(),
                             args.overlap_audit, args.output)))
