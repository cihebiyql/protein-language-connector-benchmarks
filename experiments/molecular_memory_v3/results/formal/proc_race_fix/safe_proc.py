"""Proposed future-revision /proc enumeration; imports no model/runtime code."""
from pathlib import Path


def group_members(pgid):
    members = []
    # Enumerate only the root: glob('[0-9]*/stat') can stat a disappearing PID
    # inside its iterator, before the old loop-body exception handler runs.
    # Root enumeration errors deliberately propagate: unavailable /proc is not
    # evidence that a supervised process group has no live members.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        path = entry / 'stat'
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    return members
