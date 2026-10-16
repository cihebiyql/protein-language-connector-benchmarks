"""Independent panel execution gates; these never approve a multitask claim.

Restricted external-source coverage is retained in the run scope, so unfinished
RT and external database retrieval cannot silently become zero-exposure results.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

from .contracts import ROOT, ContractError, jsonl, record_reasons, sha256


def _members(value):
    if value is None or value == '':
        return set()
    return set(value) if isinstance(value, list) else {str(value)}


def disjointness(rows, field):
    owners = defaultdict(set)
    missing = []
    for row in rows:
        value = row
        for key in field.split('.'):
            value = value.get(key) if isinstance(value, dict) else None
        members = _members(value)
        if not members:
            missing.append(row['sample_id'])
        for member in members:
            owners[member].add(row['split'])
    overlaps = {str(key): sorted(parts) for key, parts in owners.items() if len(parts) > 1}
    return {'field': field, 'missing': len(missing), 'overlapping_groups': len(overlaps),
            'examples': dict(list(overlaps.items())[:10]), 'verified': not missing and not overlaps}


def validate_panel_data(samples_path, lock_path, panel):
    samples_path, lock_path = Path(samples_path), Path(lock_path)
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if lock.get('scope') != 'panel_specific_predictor' or lock.get('panel_id') != panel:
        raise ContractError('panel-specific predictor lock required')
    if lock.get('status') not in ('frozen', 'frozen_restricted_scope'):
        raise ContractError('panel dataset has not been frozen')
    if panel == 'A1' and lock.get('scenario') not in ('new_peptide_known_target', 'new_peptide_new_target'):
        raise ContractError('A1 must identify a registered two-sided split scenario')
    pins = lock.get('pins', {})
    pinned_paths = set()
    if not pins:
        raise ContractError('physical data and audit pins required')
    for relative, expected in pins.items():
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != expected:
            raise ContractError(f'panel artifact fingerprint mismatch: {relative}')
        pinned_paths.add(path.resolve())
    if samples_path.resolve() not in pinned_paths:
        raise ContractError('training samples are not pinned by the panel lock')
    evidence = lock.get('homology_evidence', {})
    evidence_paths = [evidence.get('protocol_path'), evidence.get('group_audit_path')]
    alignments = evidence.get('alignment_paths', [])
    if not all(evidence_paths) or not alignments:
        raise ContractError('physical homology protocol, alignments and group audit required')
    for relative in evidence_paths + alignments:
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        if path.resolve() not in pinned_paths or path.stat().st_size == 0:
            raise ContractError('homology evidence is empty or not fingerprinted')
    rows = list(jsonl(samples_path))
    counts = Counter(row.get('split') for row in rows)
    if any(not counts[split] for split in ('train', 'validation', 'test')):
        raise ContractError('nonempty train, validation and test sets required')
    if set(counts) - {'train', 'validation', 'test'}:
        raise ContractError('quarantined rows cannot enter panel training')
    if len({row.get('sample_id') for row in rows}) != len(rows):
        raise ContractError('duplicate training/evaluation sample IDs')
    for row in rows:
        reasons = record_reasons(row, panel)
        if reasons:
            raise ContractError(f'ineligible row {row.get("sample_id")}: {reasons}')
    fields = (['peptide_sequence', 'groups.peptide_family', 'groups.source_doi', 'group_id'] if panel == 'A1' else
              ['segment_sequence', 'groups.parent_homology', 'groups.peptide_family', 'groups.publication', 'class_id'])
    if panel == 'A1' and lock.get('scenario') == 'new_peptide_new_target':
        fields.append('groups.target_family')
    audits = [disjointness(rows, field) for field in fields]
    if any(not audit['verified'] for audit in audits):
        raise ContractError(f'panel split leakage or unknown groups: {audits}')
    if panel == 'A1':
        components = defaultdict(set)
        for row in rows:
            keys = [('source', row['groups']['source_doi']), ('peptide', row['groups']['peptide_family'])]
            if lock['scenario'] == 'new_peptide_new_target':
                keys.append(('target', row['groups']['target_family']))
            for key in keys:
                components[key].add(row['group_id'])
        if any(len(owners) > 1 for owners in components.values()):
            raise ContractError('bootstrap group IDs split a linked source or sequence family')
    if panel == 'A1' and lock.get('scenario') == 'new_peptide_known_target':
        train = {row['groups']['target_family'] for row in rows if row['split'] == 'train'}
        heldout = {row['groups']['target_family'] for row in rows if row['split'] != 'train'}
        if not heldout <= train:
            raise ContractError('known-target scenario contains target families absent from training')
        exact_train = {row['target_sequence'] for row in rows if row['split'] == 'train'}
        if any(row['target_sequence'] not in exact_train for row in rows if row['split'] != 'train'):
            raise ContractError('known-target scenario requires exact target reference sequences observed in training')
    if panel == 'B1' and not lock.get('checks', {}).get('description_rule_reviewed'):
        raise ContractError('B1 descriptions have not been reviewed for direct answer rules')
    if panel == 'B1':
        from .prepare_elm import independent_regions
        by_class = defaultdict(list)
        for row in rows:
            by_class[row['class_id']].append(row)
        if any(independent_regions(records) < 10 for records in by_class.values()):
            raise ContractError('B1 class has fewer than ten independent parent regions')
        if any(len({row['groups']['parent_homology'] for row in records}) < 10 for records in by_class.values()):
            raise ContractError('B1 class has fewer than ten independent parent homology families')
        catalog_paths = [lock_path.parent / 'b1_classes_main.jsonl', lock_path.parent / 'b1_candidates_main.jsonl']
        if any(path.resolve() not in pinned_paths for path in catalog_paths):
            raise ContractError('both physical B1 semantic catalogs must be pinned')
        catalog = [row for row in jsonl(catalog_paths[0]) if row.get('eligible') and row.get('candidate_id')]
        candidates = list(jsonl(catalog_paths[1]))
        mapped = {row['class_id']: row for row in catalog}
        descriptions = {row['candidate_id']: row.get('description') for row in candidates}
        if len(mapped) != len(catalog) or len(descriptions) != len(candidates):
            raise ContractError('duplicate B1 class or candidate IDs')
        if {row['candidate_id'] for row in catalog} != set(descriptions):
            raise ContractError('B1 candidate universe differs between catalogs')
        for row in rows:
            item = mapped.get(row['class_id'], {})
            if item.get('split') != row['split'] or descriptions.get(item.get('candidate_id')) != row['class_description_no_rule']:
                raise ContractError('B1 gold split/description differs from the frozen anonymous catalog')
    external = lock.get('checks', {}).get('external_isolation_verified') is True
    if not external and lock['status'] != 'frozen_restricted_scope':
        raise ContractError('incomplete external coverage must retain restricted scope')
    return rows, {'panel_id': panel, 'scope': 'panel_specific_predictor',
                  'status': lock['status'], 'benchmark_lock_sha256': sha256(lock_path),
                  'samples_sha256': sha256(samples_path), 'split_counts': dict(counts),
                  'disjointness': audits, 'external_isolation_verified': external,
                  'limitations': lock.get('limitations', []),
                  'unified_multitask_connector_claim': False}
