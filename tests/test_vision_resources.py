import hashlib
import tarfile

import pytest

from mmrl.vision_resources import write_resource_archive


def asset(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {'path': name, 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


def test_archive_preserves_binary_resources_and_excludes_private_files(tmp_path):
    root = tmp_path / 'project'
    assets = [asset(root, 'resources/icons/a.png', b'\x00\xff\x80'),
              asset(root, 'resources/a b.txt', b'public')]
    asset(root, 'prototypes/target.jpg', b'private image')
    asset(root, 'workflow.json', b'private assertions')
    output = tmp_path / 'resources.tar.gz'
    assert write_resource_archive(root, assets, output) == 2
    with tarfile.open(output) as archive:
        assert archive.getnames() == [item['path'] for item in assets]
        assert all(entry.isfile() for entry in archive.getmembers())
        for item in assets:
            value = archive.extractfile(item['path']).read()
            assert hashlib.sha256(value).hexdigest() == item['sha256']


@pytest.mark.parametrize('kind', ['changed', 'duplicate', 'private', 'traversal', 'symlink'])
def test_invalid_staging_fails_before_upload(tmp_path, kind):
    root = tmp_path / 'project'
    item = asset(root, 'resources/a', b'expected')
    assets = [item]
    if kind == 'changed':
        (root / item['path']).write_bytes(b'tampered')
    elif kind == 'duplicate':
        assets.append(item)
    elif kind == 'private':
        assets = [asset(root, 'prototypes/a', b'private')]
    elif kind == 'traversal':
        item['path'] = 'resources/../secret'
    else:
        (root / item['path']).unlink()
        (root / item['path']).symlink_to(tmp_path / 'secret')
    with pytest.raises(ValueError):
        write_resource_archive(root, assets, tmp_path / 'resources.tar.gz')
