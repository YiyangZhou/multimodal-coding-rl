"""Transfer only verified public resources in one archive per project."""
import hashlib
from pathlib import PurePosixPath
import tarfile

from .vision_data import checked_file


def write_resource_archive(root, assets, destination):
    seen = set()
    with tarfile.open(destination, 'w:gz') as archive:
        for item in assets:
            name = item['path']
            parts = PurePosixPath(name).parts
            if (not name.startswith('resources/') or len(parts) < 2
                    or '..' in parts or name in seen):
                raise ValueError('Only unique public resource paths may be staged')
            path = checked_file(root, name)
            with path.open('rb') as stream:
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    size += len(block)
                    digest.update(block)
                if size != item['bytes'] or digest.hexdigest() != item['sha256']:
                    raise ValueError('Resource changed since manifest preparation')
                stream.seek(0)
                entry = tarfile.TarInfo(name)
                entry.size = size
                entry.mode = 0o644
                archive.addfile(entry, stream)
            seen.add(name)
    return len(seen)
