"""Lock restricted A1 reference-target panels without looking at label values.

Families are connected components of prespecified sequence-similarity edges.
All source DOI and peptide families stay in one partition. The double-new panel
also keeps target families in one partition; the known-target panel drops held
out rows whose exact reference target does not occur in the training partition.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AA = 'ACDEFGHIKLMNPQRSTVWY'
SCENARIOS = ('new_peptide_known_target', 'new_peptide_new_target')
PROTOCOL = {
    'version': 'A1-restricted-source-family-exhaustive-v1', 'seed': 20261015,
    'peptide': {'metric': '1-Levenshtein_distance/max_length', 'minimum': .8,
                'algorithm': 'exhaustive length-eligible pairs with exact Myers bitvector edit distance; safe composition lower bound only',
                'linkage': 'connected_components'},
    'target': {'minimum_identity': .3, 'minimum_query_coverage': .8, 'minimum_target_coverage': .8,
               'algorithm': 'MMseqs2 align all query-target pairs via official fake-prefilter, E1000; connected_components of qualifying alignments'},
    'external_targets': {'minimum_identity': .3, 'minimum_external_query_coverage': .8,
                         'minimum_candidate_target_coverage': 0., 'scope': 'conservative PEPBI template chain and Kliche reference/construct homolog exclusion'},
    'external_pepinter': 'exclude candidate peptide-family and target-family pair matching official processed affinity rows; source PMID/DOI where available',
    'source_group': 'entire normalized original DOI (broader than an experiment)',
    'nominal_fractions': {'train': .7, 'validation': .15, 'test': .15},
    'allocation': 'label-blind group size greedy target-count deficits, stable SHA256(seed,groupid) tie order',
    'known_target': 'val and test exact target_sequence SHA256 must occur in train; unsupported rows quarantined without reassignment',
    'external_missing_table': 'PEPBI actual workbook unavailable; isolate all Table1 original ITC DOIs and homologous template chains; restricted scope remains explicit',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sid(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding='utf8') if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf8')


class DSU:
    def __init__(self, keys):
        self.parent = {x: x for x in keys}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            x, self.parent[x] = self.parent[x], root
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def labels(self, prefix):
        return {key: prefix + self.find(key) for key in self.parent}


def edit_distance(a, b):
    """Exact global Levenshtein using arbitrary-width integer bit vectors."""
    if not a:
        return len(b)
    masks = {}
    for i, char in enumerate(a):
        masks[char] = masks.get(char, 0) | (1 << i)
    positive, negative, score, high = ~0, 0, len(a), 1 << (len(a) - 1)
    for char in b:
        equal = masks.get(char, 0)
        xv = equal | negative
        xh = (((equal & positive) + positive) ^ positive) | equal
        ph, mh = negative | ~(xh | positive), positive & xh
        score += bool(ph & high) - bool(mh & high)
        ph, mh = (ph << 1) | 1, mh << 1
        positive, negative = mh | ~(xv | ph), ph & xv
    return score


def composition(sequences):
    return np.array([[s.count(char) for char in AA] for s in sequences], dtype=np.int16)


def peptide_matches(queries, targets, self_pairs=False):
    """Enumerate ALL qualifying global-edit pairs; bounds cannot reject a true edge."""
    targets = list(targets)
    lengths = np.array([len(x) for x in targets])
    counts = composition(targets)
    for i, a in enumerate(queries):
        maxlen = np.maximum(len(a), lengths)
        budget = np.floor(maxlen * .2 + 1e-9).astype(int)
        lower_bound = (np.abs(counts - composition([a])[0]).sum(axis=1) + np.abs(lengths - len(a))) // 2
        indices = np.flatnonzero((np.abs(lengths - len(a)) <= budget) & (lower_bound <= budget))
        for j in indices:
            if self_pairs and j <= i:
                continue
            b = targets[j]
            distance = edit_distance(a, b)
            if distance <= budget[j]:
                yield a, b, distance, 1. - distance / max(len(a), len(b))


def fasta_records(path):
    header, parts = None, []
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if line.startswith('>'):
            if header is not None:
                yield header, ''.join(parts)
            header, parts = line[1:], []
        else:
            parts.append(line.strip())
    if header is not None:
        yield header, ''.join(parts)


def exhaustive_align_native(query, target, output, both):
    work = output.parent / ('allpairs_' + output.stem)
    work.mkdir(exist_ok=True)
    qdb, tdb, pref, aln = [work / x for x in ['queries', 'targets', 'allpairs', 'alignments']]
    commands = []
    def execute(arguments):
        commands.append(arguments)
        subprocess.run(arguments, check=True)
    execute(['mmseqs', 'createdb', str(query), str(qdb), '--dbtype', '1'])
    execute(['mmseqs', 'createdb', str(target), str(tdb), '--dbtype', '1'])
    target_index = Path(str(tdb) + '.index').read_bytes()
    pref.write_bytes(target_index)
    keys = [line.split('\t')[0] for line in Path(str(qdb) + '.index').read_text().splitlines()]
    Path(str(pref) + '.index').write_text(''.join(f'{key}\t0\t{len(target_index)}\n' for key in keys))
    Path(str(pref) + '.dbtype').write_bytes(bytes([7, 0, 0, 0]))
    execute(['mmseqs', 'align', str(qdb), str(tdb), str(pref), str(aln), '--alignment-mode', '3',
             '--min-seq-id', '0.3', '-c', '0.8', '--cov-mode', '0' if both else '2', '-e', '1000',
             '--threads', '8', '--max-accept', '2147483647', '--max-rejected', '2147483647'])
    execute(['mmseqs', 'convertalis', str(qdb), str(tdb), str(aln), str(output),
             '--format-output', 'query,target,fident,alnlen,qcov,tcov,evalue'])
    write_json(output.with_suffix('.native_commands.json'), {'commands': commands,
               'version': subprocess.check_output(['mmseqs', 'version'], text=True).strip(),
               'query_target_pairs': len(keys) * len(Path(str(tdb) + '.index').read_text().splitlines()),
               'algorithm': 'exhaustive_allpairs_fake_prefilter',
               'official_recipe': 'https://github.com/soedinglab/MMseqs2/wiki#how-to-create-a-fake-prefiltering-for-all-vs-all-alignments'})


def mmseqs(root, query, target, output, both=True):
    """All target pairs are aligned; no k-mer candidate prefilter is used."""
    arguments = ['LOCAL_HOME/.venvs/peptide-omnipanel-plm5080/bin/python', str(Path(__file__).relative_to(root)).replace('\\', '/'),
                 '--align-worker', str(query.relative_to(root)).replace('\\', '/'), str(target.relative_to(root)).replace('\\', '/'),
                 str(output.relative_to(root)).replace('\\', '/'), '--coverage-mode', '0' if both else '2']
    linux_root = '/mnt/' + root.drive[0].lower() + str(root)[2:].replace('\\', '/')
    command = 'cd ' + shlex.quote(linux_root) + ' && ' + shlex.join(arguments)
    meta = {'command': command, 'query_sha256': sha(query), 'target_sha256': sha(target),
            'coverage_mode': 'both' if both else 'external_query_only', 'algorithm': 'exhaustive_allpairs_fake_prefilter'}
    command_file = output.with_suffix('.command.json')
    if output.exists() and command_file.exists() and json.loads(command_file.read_text(encoding='utf8')) == meta:
        return
    if output.exists():
        output.rename(output.with_name(output.stem + '.previous_' + sha(output)[:12] + '.tsv'))
    write_json(command_file, meta)
    with output.with_suffix('.log').open('w', encoding='utf8') as log:
        subprocess.run(['wsl.exe', '-d', 'Ubuntu-22.04', '--', 'bash', '-lc', command], stdout=log, stderr=subprocess.STDOUT, check=True)


def load_alignments(path, require_both=True):
    result = []
    for line in Path(path).read_text().splitlines():
        a, b, identity, length, qcov, tcov, evalue = line.split('\t')
        if float(identity) >= .3 and float(qcov) >= .8 and (not require_both or float(tcov) >= .8):
            result.append({'query': a, 'target': b, 'identity': float(identity), 'aligned_length': int(length),
                           'query_coverage': float(qcov), 'target_coverage': float(tcov), 'evalue': float(evalue)})
    return result


def external_inventory(root, out):
    raw = root / 'data/raw/bench_v02/ppikb'
    pepinter = []
    for name in ['pepinter_aff_train.csv', 'pepinter_aff_test.csv']:
        with (raw / name).open(encoding='utf8') as handle:
            for i, row in enumerate(csv.DictReader(handle), 2):
                if set(row['mobile_seq']) <= set(AA) and set(row['target_seq']) <= set(AA):
                    pepinter.append({'external_id': name + ':' + str(i), 'peptide': row['mobile_seq'], 'target': row['target_seq']})
    prior = json.loads((root / 'data/processed/bench_v02/a1_external_locks.json').read_text(encoding='utf8'))
    table = json.loads((raw / 'pepbi_official/table1_source_locks.json').read_text(encoding='utf8'))
    source_dois = sorted(set(prior['locked_source_dois']) | {doi.lower() for row in table for doi in row['source_dois']})
    templates, short_peptides = [], []
    for path in sorted((raw / 'pepbi_official').glob('*.fasta')):
        for header, sequence in fasta_records(path):
            if not set(sequence) <= set(AA):
                continue
            record = {'external_id': 'pepbi:' + header, 'sequence': sequence, 'type': 'PEPBI_experimental_template_chain'}
            if len(sequence) >= 35:
                templates.append(record)
            elif 5 <= len(sequence) <= 50:
                short_peptides.append(sequence)
    for path in sorted((raw / 'kliche_external').glob('*.fasta')):
        for header, sequence in fasta_records(path):
            if len(sequence) >= 35 and set(sequence) <= set(AA):
                templates.append({'external_id': 'kliche:' + header, 'sequence': sequence, 'type': 'Kliche_target_reference'})
    catalog = json.loads((root / 'data/processed/t3_target_catalog.json').read_text(encoding='utf8'))
    for entry in catalog['entries']:
        match = re.search(r'__([A-Z0-9]+)_(\d+)_(\d+)', entry['target_id'])
        if match:
            accession, start, end = match.groups()
            path = raw / 'kliche_external' / (accession + '.fasta')
            if path.exists():
                sequence = next(fasta_records(path))[1][int(start) - 1:int(end)]
                if len(sequence) >= 35 and set(sequence) <= set(AA):
                    templates.append({'external_id': entry['target_id'], 'sequence': sequence,
                                      'type': 'Kliche_target_construct_from_official_reference_coordinates'})
    # Actual source DOI/PMID inventories make all same-publication rows ineligible,
    # even where a database represents the molecule or endpoint differently.
    pubmed = set()
    with (raw / 'pepinter_raw_PDBbind.csv').open(encoding='utf8') as handle:
        for row in csv.DictReader(handle):
            value = row.get('Pubmed ID', '').strip()
            if value and value.lower() != 'nan':
                pubmed.add(value.removesuffix('.0'))
    source_dois.extend([])
    write_json(out / 'external_inventory.json', {'source_dois': source_dois, 'pepinter_pubmed_ids': sorted(pubmed),
                                                'pepinter_pairs': pepinter, 'target_template_records': templates,
                                                'prior_revealed_peptides': prior['locked_peptide_sequences'],
                                                'pepbi_template_peptides': sorted(set(short_peptides)),
                                                'pepbi_table1_groups': table})
    return source_dois, pubmed, pepinter, templates, sorted(set(short_peptides + prior['locked_peptide_sequences']))


def allocate(rows, scenario):
    dsu = DSU(row['sample_id'] for row in rows)
    seen = {}
    for row in rows:
        keys = [('source', row['source_doi']), ('peptide', row['groups']['peptide_family'])]
        if scenario == 'new_peptide_new_target':
            keys.append(('target', row['groups']['target_family']))
        for key in keys:
            if key in seen:
                dsu.union(row['sample_id'], seen[key])
            else:
                seen[key] = row['sample_id']
    members = collections.defaultdict(list)
    for row in rows:
        members[dsu.find(row['sample_id'])].append(row)
    groups = sorted(members, key=lambda x: (-len(members[x]), sid(str(PROTOCOL['seed']) + x)))
    counts, assigned = collections.Counter(), {}
    targets = {key: fraction * len(rows) for key, fraction in PROTOCOL['nominal_fractions'].items()}
    for group in groups:
        # Count deficits are determined entirely by group sizes, never KD values.
        split = max(('train', 'validation', 'test'), key=lambda key: (targets[key] - counts[key], -('train', 'validation', 'test').index(key)))
        counts[split] += len(members[group])
        assigned[group] = split
    output, excluded = [], []
    train_targets = {row['target_sequence'] for row in rows if assigned[dsu.find(row['sample_id'])] == 'train'}
    for row in rows:
        row = dict(row)
        group = dsu.find(row['sample_id'])
        row['group_id'], row['split'] = scenario + ':' + group, assigned[group]
        row['record_reasons'] = ['reference_target_broad_assay_context_missing', 'restricted_external_table_coverage']
        if scenario == 'new_peptide_known_target' and row['split'] != 'train' and row['target_sequence'] not in train_targets:
            row['record_reasons'].append('exact_target_absent_from_train')
            row['split'] = 'quarantined'
            excluded.append(row)
        else:
            output.append(row)
    return output, excluded, sorted([len(x) for x in members.values()], reverse=True)


def verify(rows, scenario):
    splits = {split: [row for row in rows if row['split'] == split] for split in ['train', 'validation', 'test']}
    def disjoint(field):
        sets = [{row['groups'][field] for row in value} for value in splits.values()]
        return all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    checks = {'peptide_family_disjoint': disjoint('peptide_family'), 'source_disjoint': disjoint('source_doi'),
              'target_family_disjoint': disjoint('target_family'), 'duplicate_measurement_disjoint': disjoint('repeat_measurement_candidate'),
              'exact_target_seen_in_train': all(row['target_sequence'] in {r['target_sequence'] for r in splits['train']} for key in ['validation', 'test'] for row in splits[key])}
    assert checks['peptide_family_disjoint'] and checks['source_disjoint'] and checks['duplicate_measurement_disjoint']
    assert checks['target_family_disjoint'] if scenario == 'new_peptide_new_target' else checks['exact_target_seen_in_train']
    assert all(splits.values()), 'A nonempty train/val/test cannot be allocated under the frozen grouping protocol'
    return checks


def run(root=ROOT):
    out = root / 'data/processed/bench_v02/a1_locked'
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf8')) == PROTOCOL, 'Frozen protocol differs; create a new version'
    else:
        write_json(protocol_path, PROTOCOL)
    rows = read_jsonl(root / 'data/processed/bench_v02/a1_broad_candidates.jsonl')
    peptides, targets = sorted({r['peptide_sequence'] for r in rows}), sorted({r['target_sequence'] for r in rows})
    for name, sequences in [('unique_peptides', peptides), ('unique_targets', targets)]:
        (out / (name + '.fasta')).write_text(''.join('>' + sid(s) + '\n' + s + '\n' for s in sequences), encoding='utf8')
    source_dois, pubmed, pepinter, templates, holdout_peptides = external_inventory(root, out)
    peptide_dsu, target_dsu = DSU(sid(x) for x in peptides), DSU(sid(x) for x in targets)
    edges = []
    print('Clustering', len(peptides), 'peptides with exhaustive global edit comparisons', flush=True)
    for a, b, distance, identity in peptide_matches(peptides, peptides, True):
        peptide_dsu.union(sid(a), sid(b))
        edges.append({'query': sid(a), 'target': sid(b), 'edit_distance': distance, 'global_edit_identity': identity})
    write_jsonl(out / 'peptide_edges.jsonl', edges)
    print('Peptide edges', len(edges), flush=True)
    mmseqs(root, out / 'unique_targets.fasta', out / 'unique_targets.fasta', out / 'target_alignments.tsv')
    for edge in load_alignments(out / 'target_alignments.tsv'):
        target_dsu.union(edge['query'], edge['target'])
    pf, tf = peptide_dsu.labels('pf:'), target_dsu.labels('tf:')
    for row in rows:
        row['groups'].update(peptide_family=pf[sid(row['peptide_sequence'])], target_family=tf[sid(row['target_sequence'])],
                             source_doi=row['source_doi'], original_experiment='doi:' + row['source_doi'])
    write_json(out / 'group_audit.json', {'peptide_sequence_hash_to_family': pf, 'target_sequence_hash_to_family': tf,
                                        'peptide_edge_count': len(edges), 'source_experiment_rule': 'whole source DOI conservative supergroup'})
    # Map official PepInter targets to candidate target families, including exact hits.
    external_targets = sorted({row['target'] for row in pepinter} | {row['sequence'] for row in templates})
    (out / 'external_targets.fasta').write_text(''.join('>' + sid(s) + '\n' + s + '\n' for s in external_targets), encoding='utf8')
    mmseqs(root, out / 'external_targets.fasta', out / 'unique_targets.fasta', out / 'external_target_alignments.tsv', False)
    ext_to_families = collections.defaultdict(set)
    for edge in load_alignments(out / 'external_target_alignments.tsv', False):
        ext_to_families[edge['query']].add(tf[edge['target']])
    template_types = collections.defaultdict(set)
    for row in templates:
        for family in ext_to_families[sid(row['sequence'])]:
            template_types[family].add(row['type'])
    external_peptides = sorted({r['peptide'] for r in pepinter} | set(holdout_peptides))
    print('Auditing external peptide matches', len(external_peptides), flush=True)
    candidate_family_matches = collections.defaultdict(set)
    external_edges = []
    for a, b, distance, identity in peptide_matches(peptides, external_peptides):
        candidate_family_matches[pf[sid(a)]].add(b)
        external_edges.append({'candidate': sid(a), 'external': sid(b), 'edit_distance': distance, 'global_edit_identity': identity})
    write_jsonl(out / 'external_peptide_edges.jsonl', external_edges)
    pepinter_family_pairs = collections.defaultdict(list)
    for row in pepinter:
        for family in ext_to_families[sid(row['target'])]:
            pepinter_family_pairs[(row['peptide'], family)].append(row['external_id'])
    excluded, eligible, holdout_set = [], [], set(holdout_peptides)
    for row in rows:
        reasons, matches = [], []
        if row['source_doi'].lower() in source_dois:
            reasons.append('locked_original_external_itc_source_doi')
        if str(row['raw_metadata'].get('PMID', '')).removesuffix('.0') in pubmed:
            reasons.append('pepinter_source_publication_pmid')
        target_family, peptide_family = row['groups']['target_family'], row['groups']['peptide_family']
        if target_family in template_types:
            reasons.extend('external_target_homolog:' + x for x in sorted(template_types[target_family]))
        for external_peptide in candidate_family_matches[peptide_family]:
            if external_peptide in holdout_set:
                reasons.append('prior_or_pepbi_template_peptide_family')
            matches.extend(pepinter_family_pairs.get((external_peptide, target_family), []))
        if matches:
            reasons.append('pepinter_similar_molecular_pair')
        if reasons:
            excluded.append({'sample_id': row['sample_id'], 'source_doi': row['source_doi'], 'reasons': sorted(set(reasons)),
                             'external_record_matches': sorted(set(matches)), 'groups': row['groups']})
        else:
            eligible.append(row)
    write_jsonl(out / 'external_exclusion_audit.jsonl', excluded)
    # Keep all measurements and source mappings, including repeated numeric values.
    write_jsonl(out / 'measurement_provenance.jsonl', [{'sample_id': r['sample_id'], 'source_doi': r['source_doi'],
                  'source_record_id': r['source_record_id'], 'repeat_group': r['groups']['repeat_measurement_candidate'],
                  'source_table': 'Main.xlsx', 'row_policy': 'retained_or_external_quarantine_no_averaging'} for r in rows])
    print('After external quarantine', len(eligible), 'of', len(rows), flush=True)
    summary = {}
    evidence = [out / name for name in ['protocol.json', 'peptide_edges.jsonl', 'external_peptide_edges.jsonl',
                'group_audit.json', 'target_alignments.tsv', 'external_target_alignments.tsv',
                'target_alignments.command.json', 'external_target_alignments.command.json',
                'target_alignments.native_commands.json', 'external_target_alignments.native_commands.json',
                'external_inventory.json', 'external_exclusion_audit.jsonl', 'measurement_provenance.jsonl']]
    evidence += [root / 'data/raw/bench_v02/ppikb/pepbi_official' / name for name in ['article.xml', 'table1_source_locks.json', 'retrieval_attempts.json', 'pdb_fasta_fetch.json']]
    limitations = [
        'PEPBI workbook inaccessible via official endpoints (401/403); coverage limited to official Table1 ITC source DOIs and experimentally observed template sequences; full variant identity audit incomplete.',
        'Target families use exhaustive MMseqs2 all-pair alignments at >=30% identity/80% bidirectional coverage; this is an operational sequence-family definition, not all possible remote homology.',
        'No curated cross-source parent peptide coordinates or variant-series lineage; sequence connected components and full DOI grouping provide declared operational isolation.',
        'Protein inputs are database reference sequences, actual assay constructs and conditions remain unverified; these panels support broad reference-target pKD prediction only.',
        'Prior baseline results and public-source foundation-model pretraining exposure remain separate from this project supervised split audit.',
        'PepInter processed source rows have incomplete original DOI and chemistry provenance; available PMID and sequence-pair homolog matches are excluded, remaining cross-source provenance gaps retained as scope limits.',
    ]
    for scenario in SCENARIOS:
        folder = out / scenario
        folder.mkdir(exist_ok=True)
        samples, dropped, sizes = allocate(eligible, scenario)
        checks = verify(samples, scenario)
        checks['external_isolation_verified'] = False
        write_jsonl(folder / 'samples.jsonl', samples)
        write_jsonl(folder / 'test_gold.jsonl', [row for row in samples if row['split'] == 'test'])
        write_jsonl(folder / 'split_exclusion_audit.jsonl', dropped)
        counts = dict(collections.Counter(r['split'] for r in samples))
        manifest = {'panel_id': 'A1', 'scenario': scenario, 'source_rows': len(rows), 'after_external_quarantine': len(eligible),
                    'split_counts': counts, 'component_sizes': sizes, 'known_exact_target_exclusions': len(dropped),
                    'checks': checks, 'thresholds': PROTOCOL, 'no_label_stratification': True,
                    'target_reference_identity': 'exact target_sequence hash required seen for known-target evaluation',
                    'external_excluded_rows': len(excluded), 'external_exclusion_reasons': dict(collections.Counter(x for r in excluded for x in r['reasons'])),
                    'statistics_unit': 'group_id; bootstrap source/peptide-component groups; seed variability separate',
                    'limitations': limitations}
        write_json(folder / 'manifest.json', manifest)
        pins = {str(path.relative_to(root)).replace('\\', '/'): sha(path) for path in evidence + [folder / 'samples.jsonl', folder / 'test_gold.jsonl', folder / 'split_exclusion_audit.jsonl', folder / 'manifest.json', Path(__file__)]}
        lock = {'status': 'frozen_restricted_scope', 'scope': 'panel_specific_predictor', 'panel_id': 'A1', 'scenario': scenario,
                'source_rows_sha256': sha(root / 'data/processed/bench_v02/a1_broad_candidates.jsonl'),
                'split_counts': counts, 'checks': checks, 'limitations': limitations, 'pins': pins,
                'homology_evidence': {'protocol_path': str(protocol_path.relative_to(root)).replace('\\', '/'),
                                     'alignment_paths': [str((out / name).relative_to(root)).replace('\\', '/') for name in ['target_alignments.tsv', 'external_target_alignments.tsv', 'peptide_edges.jsonl', 'external_peptide_edges.jsonl']],
                                     'group_audit_path': str((out / 'group_audit.json').relative_to(root)).replace('\\', '/')}}
        write_json(folder / 'benchmark_lock.json', lock)
        summary[scenario] = {'counts': counts, 'checks': checks, 'component_sizes_top10': sizes[:10], 'dropped': len(dropped)}
    write_json(out / 'split_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--align-worker', type=Path, nargs=3)
    parser.add_argument('--coverage-mode', choices=['0', '2'], default='0')
    args = parser.parse_args()
    if args.align_worker:
        exhaustive_align_native(*args.align_worker, args.coverage_mode == '0')
    else:
        run(args.root)
