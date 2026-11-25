"""Run the pinned renderer once, then export that same scene for observability."""
import json
import runpy
from pathlib import Path

try:
    runpy.run_path('/opt/3dcodebench/render.py', run_name='__main__')
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise

# Render receipt remains authoritative; optional export must not change it.
report = {'complete': False, 'same_render_process': True}
try:
    receipt = json.loads(Path('/workspace/views/render_log.json').read_text())
    if receipt.get('status') != 'OK':
        raise ValueError('No successful final scene')
    import bpy
    bpy.ops.object.select_all(action='DESELECT')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        raise ValueError('No mesh')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.export_scene.gltf(filepath='/workspace/views/model.glb', export_format='GLB',
                              use_selection=True, export_apply=True)
    report['complete'] = True
except Exception as exc:
    report['error_type'] = type(exc).__name__
Path('/workspace/views/media-export.json').write_text(json.dumps(report))
