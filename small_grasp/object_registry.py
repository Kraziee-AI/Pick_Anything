"""CPU metadata for the isolated four-class object-single experiment.

Geometry is assembled using numeric mesh coordinates as metres in a Z-up stage.
Fruit files lack authored stage units/up-axis; their USD fallback metadata is
not an instruction to scale by .01 or rotate the mesh. Runtime physical import
and settling must be validated separately. Center offsets are scoring metadata,
never target positions for perception, grasp proposals or control.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_CONFIG_PATH = Path(__file__).with_name('object_assets.json')
_CONFIG = json.loads(ASSET_CONFIG_PATH.read_text())
OBJECT_KEYS = tuple(_CONFIG['objects'])
TABLE_TOP_Z_M = _CONFIG['table_top_z_m']
SPAWN_CLEARANCE_M = _CONFIG['spawn_clearance_m']
PHYSICS_MATERIAL = dict(_CONFIG['physics_material'])
NEWTON_CONTACT_FRICTION = dict(_CONFIG['newton_contact_friction'])


def get_object_spec(key):
    """Return independent metadata and an absolute USD path for a known key.

    ``resting_root_height_m`` is root height above the table at identity pose,
    excluding the table's world height and the initial spawn clearance.
    ``center_offset_m`` is local and already scaled; only scoring may use it.
    """
    if key not in _CONFIG['objects']:
        raise ValueError(f'Unknown object {key!r}; available: {", ".join(OBJECT_KEYS)}')
    spec = deepcopy(_CONFIG['objects'][key])
    asset = (PROJECT_ROOT / spec['asset_path']).resolve()
    if not asset.is_relative_to(PROJECT_ROOT):
        raise ValueError('Object asset must stay within Pick_Anything')
    if not asset.is_file():
        raise FileNotFoundError(f'Missing local object asset: {asset}')
    spec['usd_path'] = str(asset)
    return spec


def resolve_object_key(instruction):
    """Map one unambiguous class name; this is vocabulary, not detection."""
    text = instruction.strip().lower()
    if not text:
        raise ValueError('Target description must not be empty')
    hits = []
    for key, spec in _CONFIG['objects'].items():
        if any((alias in text if not alias.isascii() else
                re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text))
               for alias in spec['aliases']):
            hits.append(key)
    if len(hits) != 1:
        raise ValueError('Instruction must name one of orange, apple, pear or blue cube')
    return hits[0]
