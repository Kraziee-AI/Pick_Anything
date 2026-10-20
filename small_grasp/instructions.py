"""Validate the language target supported by the current physical executor."""
from pick_anything.tabletop.catalog import resolve_target


def resolve_pick_instruction(instruction):
    text = instruction.strip()
    if not text:
        raise ValueError('Target description must not be empty')
    try:
        target = resolve_target(text, available=('orange',))
    except ValueError as error:
        raise ValueError('Physical picking currently supports only one orange target; use orange or 橘子. '
                         'Other classes require their own perception, execution and scoring support.') from error
    return {'instruction': text, 'target_key': target.key,
            'grounding_instruction': target.key, 'caption': target.caption}
