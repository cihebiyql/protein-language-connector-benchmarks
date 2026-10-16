"""Prepare an auditable ELM snapshot; never turn unannotated pairs into negatives.

Only standard-library imports; no training or implicit network activity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/bench_v02/elm'
OUT = ROOT / 'data/processed/bench_v02'
SNAPSHOT = '2026-10-15'
DOI = '10.1093/nar/gkad1058'
AA = set('ACDEFGHIKLMNPQRSTVWY')
MODEL_EUKARYOTE_TAXIDS = {9606, 10090, 10116, 7227, 6239, 4932, 4896, 3702, 7955}


def read_tsv(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    metadata = [line for line in lines if line.startswith('#')]
    body = '\n'.join(line for line in lines if not line.startswith('#'))
    return list(csv.DictReader(io.StringIO(body), delimiter='\t')), metadata


def read_fasta(path):
    sequences, accession, parts = {}, None, []
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if line.startswith('>'):
            if accession:
                seq = ''.join(parts)
                if accession in sequences and sequences[accession] != seq:
                    raise ValueError(f'Conflicting FASTA for {accession}')
                sequences[accession] = seq
            header = line[1:].split()[0].split('|')
            accession = header[1] if len(header) >= 3 else header[0]
            parts = []
        elif line.strip():
            parts.append(line.strip().upper())
    if accession:
        seq = ''.join(parts)
        if accession in sequences and sequences[accession] != seq:
            raise ValueError(f'Conflicting FASTA for {accession}')
        sequences[accession] = seq
    return sequences


def clean_description(text):
    """Redact explicit sequence syntax; preserve the original in the rule panel.

    Conservative token removal can also mask acronyms. Human description review
    remains a publication gate; this is not an assertion of perfect leakage removal.
    """
    direct = re.compile(r'\[[A-Z^]+\]|\{\d|[ACDEFGHIKLMNPQRSTVWYXx]{3,}|(?:\b[A-ZxX][-|]){2}|\b(?:regex|consensus|pattern)\b')
    text = re.sub(r'\([^)]*(?:\[[A-Z^]+\]|(?:[A-ZxX][-|]){2}|[ACDEFGHIKLMNPQRSTVWYXx]{3,})[^)]*\)', '[rule omitted]', text)
    return direct.sub('[rule omitted]', text).strip()


def class_split(class_id, scope='human'):
    if scope == 'model_eukaryotes':
        value = int(hashlib.sha256(('pepbridge-v02-class-model_eukaryotes-70_15_15-v1-20261015:' + class_id).encode()).hexdigest()[:8], 16) % 20
        return 'test' if value < 3 else ('validation' if value < 6 else 'train')
    if scope != 'human':
        raise ValueError(f'Unknown ELM class split scope: {scope}')
    value = int(hashlib.sha256(('pepbridge-v02-class-20261015:' + class_id).encode()).hexdigest()[:8], 16) % 10
    return 'test' if value < 2 else ('validation' if value == 2 else 'train')


def independent_regions(rows):
    """Count connected overlapping intervals once within each parent."""
    parents = defaultdict(list)
    for row in rows:
        parents[row['parent_accession']].append((row['start'], row['end']))
    total = 0
    for intervals in parents.values():
        last_end = None
        for start, end in sorted(set(intervals)):
            if last_end is None or start > last_end:
                total += 1
                last_end = end
            else:
                last_end = max(last_end, end)
    return total


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf8')


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows), encoding='utf8')


def external_exclusions():
    parents, sequences = set(), set()
    files = ['benz_samples.jsonl', 't3_d2_pairs.jsonl', 't3_g1_records.jsonl']
    for name in files:
        path = ROOT / 'data/processed' / name
        if not path.exists():
            raise FileNotFoundError(f'Cross-database audit requires {path}')
        for line in path.read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            parents.update(p['accession'].split('-')[0] for p in row.get('parents', []))
            parents.update(p.split('-')[0] for p in row.get('parent_accessions', []))
            if row.get('parent_accession'):
                parents.add(row['parent_accession'].split('-')[0])
            sequences.update(row[k] for k in ['sequence', 'sequence_a', 'sequence_b'] if row.get(k))
    return parents, sequences, files


def verified_taxonomy(raw, allowed_taxids=MODEL_EUKARYOTE_TAXIDS):
    """Exact official names/aliases only, with verified allowed lineage for strains."""
    names = {}
    found = set()
    for path in sorted(Path(raw).glob('taxonomy_*.json')):
        record = json.loads(path.read_text(encoding='utf8'))
        taxid = record.get('taxonId')
        if not record.get('active', False) or not taxid:
            continue
        lineage = {x['taxonId'] for x in record.get('lineage', [])}
        ancestors = ({taxid} | lineage) & set(allowed_taxids)
        if not ancestors:
            continue
        if taxid in allowed_taxids:
            found.add(taxid)
        info = {'taxid': taxid, 'allowed_species_taxid': min(ancestors),
                'taxonomy_source': f'https://rest.uniprot.org/taxonomy/{taxid}.json',
                'taxonomy_raw_file': path.name}
        for name in [record['scientificName']] + record.get('otherNames', []):
            key = name.casefold()
            if key in names and names[key]['taxid'] != taxid:
                raise ValueError(f'Ambiguous taxonomy alias: {name}')
            names[key] = info
    if found != set(allowed_taxids):
        raise ValueError(f'Missing verified species taxonomy: {sorted(set(allowed_taxids) - found)}')
    return names


def prepare(raw=RAW, out=OUT, scope='human'):
    if scope not in {'human', 'model_eukaryotes'}:
        raise ValueError(f'Unknown ELM scope: {scope}')
    if scope == 'model_eukaryotes' and out.resolve() == OUT.resolve():
        raise ValueError('model_eukaryotes requires its own output directory; preserve human snapshot')
    taxonomy = verified_taxonomy(raw) if scope == 'model_eukaryotes' else None
    classes, class_meta = read_tsv(raw / 'classes.tsv')
    instances, instance_meta = read_tsv(raw / 'instances.tsv')
    fasta = read_fasta(raw / 'instances.fasta')
    if len(instances) <= 100:
        raise ValueError('ELM default endpoint is limited to 100 records; need explicit all-instance query')
    catalog = {c['ELMIdentifier']: c for c in classes}
    external_parents, external_sequences, external_files = external_exclusions()
    normalized, evidence, class_rows = [], [], []
    for index, c in enumerate(sorted(classes, key=lambda x: x['ELMIdentifier'])):
        desc = clean_description(c['Description']) or clean_description(c['FunctionalSiteName'])
        class_rows.append({'class_id': c['ELMIdentifier'], 'candidate_id': f'candidate_{index:04d}',
                           'class_description_no_rule': desc, 'source_refs': ['http://elm.eu.org/elms/' + c['ELMIdentifier']],
                           'split': class_split(c['ELMIdentifier'], scope), 'description_review_required': True})
    class_lookup = {c['class_id']: c for c in class_rows}
    for inst in instances:
        accession, class_id = inst['Primary_Acc'], inst['ELMIdentifier']
        start, end = int(inst['Start']), int(inst['End'])
        seq = fasta.get(accession)
        reasons = []
        if inst['InstanceLogic'].strip().lower() != 'true positive':
            reasons.append('not_curated_true_positive')
        if not seq:
            reasons.append('missing_official_parent_fasta')
        elif not (1 <= start <= end <= len(seq)):
            reasons.append('invalid_parent_coordinates')
        segment = seq[start-1:end] if seq and 1 <= start <= end <= len(seq) else None
        if segment and not set(segment) <= AA:
            reasons.append('noncanonical_residue')
        if accession.split('-')[0] in external_parents:
            reasons.append('cross_database_parent_Benz_or_Kliche')
        if segment and any(segment in x or x in segment for x in external_sequences if len(x) >= 6):
            reasons.append('cross_database_segment_Benz_or_Kliche')
        if class_id not in class_lookup or not class_lookup[class_id]['class_description_no_rule']:
            reasons.append('missing_rule_free_description')
        taxon_info = taxonomy.get(inst['Organism'].casefold()) if taxonomy is not None else None
        if scope == 'human' and inst['Organism'] != 'Homo sapiens':
            reasons.append('outside_human_local_region_scope')
        elif scope == 'model_eukaryotes' and not taxon_info:
            reasons.append('outside_verified_model_eukaryotes_scope')
        pmids = inst['References'].split()
        if set(pmids) & {'35044719', '39009827'}:
            reasons.append('locked_Benz_or_Kliche_experiment_publication')
        row = {'sample_id': 'elm_' + inst['Accession'], 'panel_id': 'B1', 'task': 'unseen_function', 'elm_instance_id': inst['Accession'],
               'segment_sequence': segment, 'parent_accession': accession, 'start': start, 'end': end,
               'coordinate_system': '1-based-inclusive', 'class_id': class_id,
               'class_description_no_rule': class_lookup.get(class_id, {}).get('class_description_no_rule'),
               'source_doi': DOI, 'source_refs': ['https://pubmed.ncbi.nlm.nih.gov/' + p + '/' for p in pmids],
               'source_raw_row': inst, 'groups': {'parent': accession.split('-')[0], 'region': f'{accession}:{start}-{end}',
                                               'class': class_id, 'publication': pmids,
                                               'sequence': hashlib.sha256(segment.encode()).hexdigest() if segment else None},
               'split': class_split(class_id, scope), 'readiness_reasons': reasons,
               'eligibility': {'sequence_verified': bool(segment and set(segment) <= AA), 'no_rule_prompt': True},
               'molecule_scope': 'local_functional_region_in_parent_protein', 'free_peptide_activity_claim': False}
        if scope == 'model_eukaryotes':
            row['taxon'] = taxon_info
            row['organism_as_reported'] = inst['Organism']
            row['eligibility']['taxonomy_verified'] = bool(taxon_info)
        normalized.append(row)
        evidence.append({'evidence_id': row['sample_id'], 'panel_id': 'B2', 'task': 'knowledge_composition', 'molecular_fact': {'kind': 'curated_local_function_instance',
                         'parent_accession': accession, 'start': start, 'end': end, 'class_id': class_id,
                         'instance_logic': inst['InstanceLogic'], 'methods_as_reported': inst['Methods']},
                         'class_background_ref': class_id, 'source_refs': row['source_refs'], 'groups': row['groups'],
                         'review_status': 'review_pending', 'qa_eligible': False,
                         'required_review': ['verify_primary_experiment', 'select_independent_background_relation',
                                             'check_answer_leakage', 'define_joint_answer_and_inference_boundary',
                                             'lock_relation_question_combination_holdout'],
                         'structural_link_is_functional_hotspot': False, 'pdb_links_as_reported': inst['PDB'],
                         'readiness_reasons': reasons})
    eligible = [r for r in normalized if not r['readiness_reasons']]
    # Hold out classes first; quarantine parent/sequence/publication groups crossing
    # that fixed boundary rather than moving a held-out class into training.
    boundaries = defaultdict(set)
    for r in eligible:
        keys = ['parent:' + r['groups']['parent'], 'sequence:' + r['segment_sequence']]
        keys += ['pubmed:' + p for p in r['groups']['publication']]
        for key in keys:
            boundaries[key].add(r['split'])
    for r in eligible:
        keys = ['parent:' + r['groups']['parent'], 'sequence:' + r['segment_sequence']]
        keys += ['pubmed:' + p for p in r['groups']['publication']]
        if any(len(boundaries[key]) > 1 for key in keys):
            r['readiness_reasons'].append('class_holdout_crosses_parent_sequence_or_publication')
    grouped = defaultdict(list)
    for r in normalized:
        if not r['readiness_reasons']:
            grouped[r['class_id']].append(r)
    audits = []
    for c in class_rows:
        regions = independent_regions(grouped[c['class_id']])
        c['eligible'] = regions >= 10
        audits.append({'class_id': c['class_id'], 'split': c['split'], 'usable_records': len(grouped[c['class_id']]),
                       'independent_parent_regions': regions, 'independent_parents': len({r['groups']['parent'] for r in grouped[c['class_id']]}),
                       'minimum_independent_regions': 10, 'eligible': regions >= 10})
    final = []
    for r in normalized:
        if not class_lookup.get(r['class_id'], {}).get('eligible', False):
            r['readiness_reasons'].append('class_below_10_independent_regions')
        if not r['readiness_reasons']:
            final.append({k: v for k, v in r.items() if k != 'source_raw_row'})
        else:
            r['assigned_class_split'] = r['split']
            r['split'] = 'quarantine'
    # Regex and full raw description are intentionally a separate panel.
    rules = [{'class_id': c['ELMIdentifier'], 'regex': c['Regex'], 'description_full': c['Description'],
              'panel_id': 'B1_regex_full', 'baseline_required': 'ELM_rule_match'} for c in classes]
    write_jsonl(out / 'b1_instance_audit.jsonl', normalized)
    write_jsonl(out / 'b1_samples.jsonl', final)
    write_jsonl(out / 'b1_model_inputs.jsonl', [{'sample_id': r['sample_id'], 'panel_id': 'B1', 'task': 'unseen_function',
                    'molecule_scope': r['molecule_scope'], 'segment_sequence': r['segment_sequence'],
                    'candidate_catalog': 'b1_candidates_main.jsonl'} for r in final])
    write_jsonl(out / 'b1_candidates_main.jsonl', [{'candidate_id': c['candidate_id'],
                    'description': c['class_description_no_rule']} for c in class_rows if c['eligible']])
    write_jsonl(out / 'b1_classes_main.jsonl', class_rows)
    write_jsonl(out / 'b1_classes_regex_full.jsonl', rules)
    write_jsonl(out / 'b1_class_eligibility.jsonl', audits)
    write_jsonl(out / 'b2_evidence_registry.jsonl', evidence)
    summary = {'snapshot_date': SNAPSHOT, 'version_scope': 'official_current_snapshot_not_2024_archive',
               'paper_2024_reported': {'classes': 356, 'instances': 4283}, 'downloaded_classes': len(classes),
               'downloaded_instance_records': len(instances), 'official_fasta_parents': len(fasta),
               'raw_curated_true_positive': sum(r['InstanceLogic'] == 'true positive' for r in instances),
               'eligible_classes': sum(a['eligible'] for a in audits), 'eligible_samples': len(final),
               'eligible_split_counts': dict(Counter(r['split'] for r in final)),
               'eligible_split_class_counts': dict(Counter(a['split'] for a in audits if a['eligible'])),
               'class_split_rule': 'SHA256 pepbridge-v02-class-20261015 + class_id, modulo10: 0/1=test,2=validation,3..9=train',
               'label_class_holdout': True, 'homology_cluster_audit': 'pending_global_cross_source_audit',
               'description_manual_review': 'pending', 'main_candidate_universe': 'all_eligible_classes_seen_plus_unseen',
               'eligible_unique_sequences': len({r['segment_sequence'] for r in final}),
               'exclusion_reason_counts': dict(Counter(reason for r in normalized for reason in r['readiness_reasons'])),
               'class_download_metadata': class_meta, 'instance_download_metadata': instance_meta,
               'cross_database_input_files': external_files, 'external_parent_exclusions': len(external_parents),
               'b2_evidence_cards': len(evidence), 'b2_reviewed_qa': 0,
               'readiness': 'audit_ready_training_requires_global_source_and_description_review',
               'c2_scope': 'PDB cross-reference is structural provenance only; no inferred contact or functional hotspot label'}
    if scope == 'model_eukaryotes':
        summary.update({'scope': scope, 'scope_selection_basis': 'prespecified nonpathogen model-organism whitelist; no model performance used',
                        'class_split_rule': 'SHA256 pepbridge-v02-class-model_eukaryotes-70_15_15-v1-20261015: + class_id, modulo20: 0..2=test,3..5=validation,6..19=train',
                        'nominal_class_split_proportions': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
                        'previous_prepared_rule_snapshot': 'prepared_rule_70_10_20/',
                        'allowed_species_taxids': sorted(MODEL_EUKARYOTE_TAXIDS),
                        'taxonomy_resolution': 'exact case-insensitive official scientific name or alias; verified strain lineage accepted',
                        'eligible_organism_counts': dict(Counter(r['organism_as_reported'] for r in final)),
                        'eligible_taxid_counts': dict(Counter(str(r['taxon']['taxid']) for r in final)),
                        'scope_curated_true_positive_records': sum(r['source_raw_row']['InstanceLogic'] == 'true positive' and r['taxon'] is not None for r in normalized)})
    write_json(out / 'b1_manifest.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--scope', choices=['human', 'model_eukaryotes'], default='human')
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.out, args.scope), ensure_ascii=False, indent=2))
