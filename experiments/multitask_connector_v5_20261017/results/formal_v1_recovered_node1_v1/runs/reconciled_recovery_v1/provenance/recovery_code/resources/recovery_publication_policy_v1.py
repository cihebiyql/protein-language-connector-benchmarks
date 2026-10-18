"""Keep the original administrative ledger private; publish its SHA and derived history."""
from contextlib import contextmanager
from pathlib import PurePosixPath
import sync_results

@contextmanager
def policy():
    original = sync_results.publication_restriction
    def restriction(relative):
        if PurePosixPath(relative).as_posix() == 'runs/reconciled_recovery_v1/provenance/original_queue_events.jsonl':
            return ('Original administrative ledger retained byte-exact in the private export; '
                    'source SHA recorded here; all historical states remain in the normalized derived queue ledger')
        return original(relative)
    sync_results.publication_restriction = restriction
    try:
        yield
    finally:
        sync_results.publication_restriction = original
