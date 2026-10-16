"""Audit recovered official PEPBI and create additive A1 v2 isolation artifacts.

Never edit an existing A1 lock, sample or result. Existing split assignments are
retained; only conservative external-family exclusions can remove records.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
import unicodedata

import openpyxl

from scripts.bench_v02 import split_affinity as split

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = '0f3ae5e9021b5c0e925114c5b53a0cb441de90adc42134e5144d4ac9c4826a24'


def normalized_name(text):
    return ''.join(char for char in unicodedata.normalize('NFKC', text).lower() if char.isalnum())


def run(root=ROOT):
    recovery = root / 'data/raw/bench_v02/ppikb/pepbi_official/browser_recovery'
    workbook_path = recovery / 'PEPBI.xlsx'
    assert workbook_path.stat().st_size == 217930 and split.sha(workbook_path) == EXPECTED
    previous = root / 'data/processed/bench_v02/a1_locked'
    output = root / 'data/processed/bench_v02/a1_locked_v2'
    output.mkdir(exist_ok=True)
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    iterator = workbook.active.iter_rows(values_only=True)
    next(iterator)
    headers = next(iterator)
    original = [dict(zip(headers, row)) for row in iterator]
    workbook.close()
    table_path = recovery.parent / 'table1_source_locks.json'
    table = json.loads(table_path.read_text(encoding='utf8'))
    source_by_group = {normalized_name(row['binding_group']): row for row in table}
    inventory = []
    for index, row in enumerate(original, 3):
        source = source_by_group[normalized_name(row['Binding Group'])]
        assert source['source_dois']
        kd = row['KD (M)']
        assert isinstance(kd, (float, int)) and kd > 0
        inventory.append({'sample_id': row['PEPBI Complex Name'], 'source_excel_row': index,
                          'binding_group': row['Binding Group'], 'experimental_template_pdb': row['PDB ID'],
                          'crystallographic_unit': row['Crystallographic Unit'], 'unit_copy': row['Unit Copy'],
                          'peptide_sequence': row['Peptide Sequence'], 'target_sequence': row['Protein Sequence'],
                          'target_tag_as_reported': row['Protein Tag'], 'target_mutations': row['Protein Mutation(s)'],
                          'peptide_mutations': row['Peptide Mutation(s)'],
                          'kd_molar': kd, 'pkd': -math.log10(kd), 'reported_kd_error_molar': row['KD SD (M)'],
                          'temperature_kelvin': row['T (K)'],
                          'chemical_state': {'residues': 'all canonical uppercase amino acids as supplied',
                                             'n_terminal': None, 'c_terminal': None, 'topology': None,
                                             'source_field_coverage': 'workbook has no explicit peptide terminal state or cyclization columns'},
                          'original_itc_source_dois': source['source_dois'],
                          'source_doi_assignment': 'exact normalized Binding Group join to official article Table1 ITC references',
                          'raw_metadata': row})
    inventory_path = output / 'pepbi_complete_sample_inventory.jsonl'
    split.write_jsonl(inventory_path, inventory)
    candidate_rows = split.read_jsonl(root / 'data/processed/bench_v02/a1_broad_candidates.jsonl')
    graph = json.loads((previous / 'group_audit.json').read_text(encoding='utf8'))
    peptide_families = graph['peptide_sequence_hash_to_family']
    target_families = graph['target_sequence_hash_to_family']
    peptides = sorted({r['peptide_sequence'] for r in candidate_rows})
    external_peptides = sorted({r['peptide_sequence'] for r in inventory})
    peptide_family_matches = collections.defaultdict(set)
    peptide_edges = []
    for a, b, distance, identity in split.peptide_matches(peptides, external_peptides):
        family = peptide_families[split.sid(a)]
        peptide_family_matches[family].add(b)
        peptide_edges.append({'candidate': split.sid(a), 'external': split.sid(b), 'family': family,
                              'edit_distance': distance, 'global_edit_identity': identity})
    edges_path = output / 'pepbi_actual_peptide_edges.jsonl'
    split.write_jsonl(edges_path, peptide_edges)
    # Already executed all-pair actual construct→reference alignments use exactly
    # the v1 external-query coverage/identity thresholds. Reverse alignments are
    # saved as diagnostics and are not silently substituted for this protocol.
    actual_target_matches = collections.defaultdict(set)
    for edge in split.load_alignments(recovery / 'actual_target_to_candidate.tsv', False):
        actual_target_matches[target_families[edge['target']]].add(edge['query'])
    source_dois = {doi.lower() for row in inventory for doi in row['original_itc_source_dois']}
    global_exclusions = []
    old_excluded_ids = {r['sample_id'] for r in split.read_jsonl(previous / 'external_exclusion_audit.jsonl')}
    for row in candidate_rows:
        pf = peptide_families[split.sid(row['peptide_sequence'])]
        tf = target_families[split.sid(row['target_sequence'])]
        peptide_hits = peptide_family_matches.get(pf, set())
        target_hits = actual_target_matches.get(tf, set())
        source_hit = row['source_doi'].lower() in source_dois
        if peptide_hits or target_hits or source_hit:
            global_exclusions.append({'sample_id': row['sample_id'], 'source_doi': row['source_doi'],
                                      'peptide_family': pf, 'target_family': tf,
                                      'peptide_family_matches': sorted(peptide_hits), 'actual_target_matches': sorted(target_hits),
                                      'original_source_doi_match': source_hit, 'already_quarantined_v1': row['sample_id'] in old_excluded_ids})
    global_exclusions_path = output / 'pepbi_actual_global_broad_exclusion_audit.jsonl'
    split.write_jsonl(global_exclusions_path, global_exclusions)
    protocol = {'version': 'A1-PEPBI-actual-inventory-supplement-v2',
                'parent_protocol_sha256': split.sha(previous / 'protocol.json'),
                'peptide_metric': split.PROTOCOL['peptide'], 'external_target_metric': split.PROTOCOL['external_targets'],
                'exclusion_rule': 'whole original ITC DOI OR candidate peptide family matching any actual PEPBI peptide OR candidate target family matching any actual PEPBI construct',
                'existing_partitions_retained': True, 'label_used_for_partition_or_exclusion': False,
                'allocation_rationale': 'Preserve the pre-existing label-blind source/peptide/target component allocation; apply new external-family quarantine to the entire original5122-row broad pool, then filter each old scenario and recheck exact known targets. No evaluation cases are reselected after inspecting model scores.',
                'chemical_state_policy': 'missing terminal chemistry cannot establish different molecules; conservative sequence-family quarantine applies regardless of missing chemical state'}
    protocol_path = output / 'coverage_protocol.json'
    split.write_json(protocol_path, protocol)
    coverage = {}
    for scenario in split.SCENARIOS:
        parent = previous / scenario
        folder = output / scenario
        folder.mkdir(exist_ok=True)
        rows, excluded = [], []
        for row in split.read_jsonl(parent / 'samples.jsonl'):
            pf, tf = row['groups']['peptide_family'], row['groups']['target_family']
            peptide_hits, target_hits = peptide_family_matches.get(pf, set()), actual_target_matches.get(tf, set())
            source_hit = row['source_doi'].lower() in source_dois
            if peptide_hits or target_hits or source_hit:
                pair = [x['sample_id'] for x in inventory if x['peptide_sequence'] in peptide_hits and split.sid(x['target_sequence']) in target_hits]
                excluded.append({'sample_id': row['sample_id'], 'split': row['split'], 'source_doi': row['source_doi'],
                                 'peptide_family_matches': sorted(peptide_hits), 'actual_target_matches': sorted(target_hits),
                                 'original_source_doi_match': source_hit, 'similar_pair_matches': pair,
                                 'reason': 'additional_actual_PEPBI_external_family_quarantine'})
            else:
                rows.append(row)
        # Preserve the known-target meaning if source quarantine removed its only
        # training observation; quarantine held-out rows instead of moving them.
        extra = []
        if scenario == 'new_peptide_known_target':
            training_targets = {r['target_sequence'] for r in rows if r['split'] == 'train'}
            extra = [r for r in rows if r['split'] != 'train' and r['target_sequence'] not in training_targets]
            rows = [r for r in rows if r not in extra]
        checks = split.verify(rows, scenario)
        checks.update(external_isolation_verified=False, pepbi_complete_inventory_audited=True,
                      pepbi_actual_source_and_sequence_family_isolation_verified=True)
        split.write_jsonl(folder / 'samples.jsonl', rows)
        split.write_jsonl(folder / 'test_gold.jsonl', [r for r in rows if r['split'] == 'test'])
        split.write_jsonl(folder / 'new_external_exclusions.jsonl', excluded)
        split.write_jsonl(folder / 'known_target_dependency_exclusions.jsonl', extra)
        old_lock = json.loads((parent / 'benchmark_lock.json').read_text(encoding='utf8'))
        limitations = [x for x in old_lock['limitations'] if 'PEPBI workbook inaccessible' not in x]
        limitations.append('PEPBI actual workbook covers all329 rows with actual target/peptide sequences and source-DOI join; terminal chemistry lacks explicit fields, so exclusion conservatively ignores unknown chemical distinctions.')
        manifest = {'panel_id': 'A1', 'scenario': scenario, 'version': protocol['version'],
                    'parent_lock_sha256': split.sha(parent / 'benchmark_lock.json'),
                    'parent_samples_sha256': split.sha(parent / 'samples.jsonl'), 'parent_test_gold_sha256': split.sha(parent / 'test_gold.jsonl'),
                    'pepbi_workbook_sha256': EXPECTED, 'pepbi_rows': len(inventory),
                    'source_doi_join_missing_rows': 0, 'unique_peptides': len(external_peptides),
                    'unique_actual_targets': len({r['target_sequence'] for r in inventory}), 'unique_original_itc_dois': len(source_dois),
                    'new_external_excluded_rows': len(excluded), 'additional_known_target_dependency_exclusions': len(extra),
                    'whole_broad_pool_rows': len(candidate_rows), 'whole_broad_actual_pepbi_matches': len(global_exclusions),
                    'new_whole_pool_quarantine_rows': sum(not r['already_quarantined_v1'] for r in global_exclusions),
                    'prior_allocation_preserved': True,
                    'new_exclusion_splits': dict(collections.Counter(r['split'] for r in excluded)),
                    'split_counts': dict(collections.Counter(r['split'] for r in rows)), 'checks': checks, 'limitations': limitations}
        split.write_json(folder / 'manifest.json', manifest)
        lock = dict(old_lock)
        lock.update(status='frozen_restricted_scope', version=protocol['version'], checks=checks, limitations=limitations,
                    split_counts=manifest['split_counts'], parent_lock_sha256=manifest['parent_lock_sha256'],
                    parent_samples_sha256=manifest['parent_samples_sha256'], data_changed_by_external_quarantine=True)
        pins = dict(old_lock['pins'])
        additions = [workbook_path, Path(__file__), inventory_path, protocol_path, edges_path, table_path, global_exclusions_path,
                     parent / 'benchmark_lock.json', recovery / 'browser_retrieval_manifest.json',
                     recovery / 'actual_target_to_candidate.tsv', recovery / 'actual_target_to_candidate.command.json',
                     recovery / 'actual_target_to_candidate.native_commands.json',
                     folder / 'samples.jsonl', folder / 'test_gold.jsonl', folder / 'new_external_exclusions.jsonl',
                     folder / 'known_target_dependency_exclusions.jsonl', folder / 'manifest.json']
        pins.update({str(path.relative_to(root)).replace('\\', '/'): split.sha(path) for path in additions})
        lock['pins'] = pins
        lock['homology_evidence'] = {'protocol_path': str(protocol_path.relative_to(root)).replace('\\', '/'),
                                    'group_audit_path': old_lock['homology_evidence']['group_audit_path'],
                                    'alignment_paths': old_lock['homology_evidence']['alignment_paths'] + [str((recovery / 'actual_target_to_candidate.tsv').relative_to(root)).replace('\\', '/'), str(edges_path.relative_to(root)).replace('\\', '/')]}
        split.write_json(folder / 'benchmark_lock.json', lock)
        coverage[scenario] = manifest
    split.write_json(output / 'coverage_supplement.json', coverage)
    print(json.dumps({scenario: {'split_counts': data['split_counts'], 'new_exclusions': data['new_external_excluded_rows']} for scenario, data in coverage.items()}, indent=2))


if __name__ == '__main__':
    run()
