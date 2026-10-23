"""Validate the language target supported by the current physical executor."""
from pick_anything.tabletop.catalog import resolve_target


def resolve_pick_instruction(instruction, *, allow_objects=False):
    text = instruction.strip()
    if not text:
        raise ValueError('Target description must not be empty')
    if allow_objects:
        from .object_registry import get_object_spec, resolve_object_key
        key = resolve_object_key(text)
        spec = get_object_spec(key)
        return {'instruction': text, 'target_key': key,
                'grounding_instruction': spec['caption'], 'caption': spec['caption']}
    try:
        target = resolve_target(text, available=('orange',))
    except ValueError as error:
        raise ValueError('Physical picking currently supports only one orange target; use orange or 橘子. '
                         'Other classes require their own perception, execution and scoring support.') from error
    return {'instruction': text, 'target_key': target.key,
            'grounding_instruction': target.key, 'caption': target.caption}
