"""Lock B1 with exhaustive MMseqs2 alignments and explicit evidence boundaries.

No new dependencies. Run prepare/build on Windows or all stages under existing WSL.
The all-pairs alignment route follows the official MMseqs2 fake-prefilter recipe.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('prepare_elm', Path(__file__).with_name('prepare_elm.py'))
elm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(elm)
INPUT = ROOT / 'data/processed/bench_v02/b1_model_eukaryotes'
OUT = ROOT / 'data/processed/bench_v02/b1_homology_locked'
PARAMETERS = {'parent': {'identity': .30, 'query_coverage': .80, 'target_coverage': .80, 'evalue': 1e-5},
              'context': {'identity': .50, 'query_coverage': .80, 'target_coverage': .80, 'evalue': 1e-5,
                          'minimum_alignment_length': 30, 'flank_residues_each_side': 32},
              'class_rule': 'unchanged model_eukaryotes-70_15_15-v1',
              'action': 'quarantine entire connected component crossing split or any resolved external parent/context'}

# Per-class review against the downloaded original ELM Description field.
# Wording keeps functional relation/partner semantics and removes residue rules,
# explicit motif sequences, positional preferences and examples of the answer.
REVIEWED_DESCRIPTIONS = {
 'DEG_SCF_TRCP1_1': 'Phosphorylation-dependent degron recognized by the beta-TrCP-containing SCF ubiquitin ligase complex, promoting substrate degradation.',
 'DOC_CYCLIN_RxL_1': 'Docking site recognized by S-phase cyclin and cyclin-dependent kinase complexes in fungal and mammalian target proteins.',
 'DOC_PP1_RVXF_1': 'Docking motif for the catalytic subunit of protein phosphatase 1, enabling substrate targeting for dephosphorylation.',
 'DOC_WW_Pin1_4': 'Interaction site recognized by the WW domain of the phosphorylation-dependent prolyl isomerase Pin1.',
 'LIG_14-3-3_CanoR_1': 'Canonical phosphorylation-dependent interaction motif for 14-3-3 proteins.',
 'LIG_ARS2_EDGEI_1': 'Interaction motif mediating binding to the C-terminal leg domain of ARS2 in proteins involved in RNA decay, processing or splicing.',
 'LIG_BH_BH3_1': 'Interaction motif in pro-apoptotic proteins that binds anti-apoptotic Bcl-2 family members to regulate apoptosis.',
 'LIG_CaM_IQ_9': 'Helical interaction motif responsible for calcium-independent binding to calmodulin.',
 'LIG_CtBP_PxDLS_1': 'Interaction motif for the NAD-dependent transcriptional corepressor CtBP.',
 'LIG_EH1_1': 'Repressor-associated interaction motif enabling recruitment of Groucho/TLE corepressors by homeodomain-containing and other transcription factors.',
 'LIG_HCF-1_HBM_1': 'Interaction motif for the N-terminal kelch propeller domain of the cell-cycle regulator host cell factor 1.',
 'LIG_LIR_Gen_1': 'Canonical interaction motif for Atg8/LC3 family proteins, mediating processes involved in autophagy.',
 'LIG_PCNA_PIPBox_1': 'Canonical PCNA-binding motif found in proteins involved in DNA replication, repair, methylation and cell-cycle control.',
 'LIG_PDZ_Class_1': 'C-terminal interaction motif for class 1 PDZ-domain recognition.',
 'LIG_SH3_CIN85_PxpxPR_1': 'Noncanonical SH3-binding motif recognized primarily by CIN85 and CD2AP, associated with endocytosis, lysosomal degradation, cytoskeletal regulation and signaling.',
 'LIG_SUMO_SIM_par_1': 'Interaction motif supporting the parallel beta-augmentation mode of noncovalent SUMO binding.',
 'MOD_CDK_SPxxK_3': 'Extended variant of a cyclin-dependent kinase phosphorylation site.',
 'MOD_CK1_1': 'Casein kinase 1 phosphorylation site.',
 'MOD_CK2_1': 'Casein kinase 2 phosphorylation site.',
 'MOD_N-GLC_1': 'Generic site for N-linked glycosylation.',
 'MOD_NMyristoyl': 'Generic site for N-terminal myristoylation.',
 'MOD_PIKK_1': 'Phosphorylation site recognized by phosphatidylinositol-3-kinase-related protein kinase family members.',
 'MOD_PKA_1': 'Primary-preference phosphorylation site recognized by PKA-type AGC kinases.',
 'MOD_PKA_2': 'Secondary-preference phosphorylation site recognized by PKA-type AGC kinases.',
 'MOD_Plk_1': 'Phosphorylation site recognized by polo-like kinase 1.',
 'MOD_ProDKin_1': 'Phosphorylation site recognized by the proline-directed kinase group in higher eukaryotes.',
 'MOD_SUMO_for_1': 'Conventional motif recognized for modification by SUMO-1.',
 'MOD_SUMO_rev_2': 'Inverted-orientation motif recognized for modification by SUMO-1.',
 'TRG_DiLeu_BaEn_1': 'Sorting motif recognized by adaptin sigma subunits for endosomal and basolateral trafficking.',
 'TRG_DiLeu_LyEn_5': 'Sorting motif interacting with AP-3 for endosomal and lysosomal trafficking.',
 'TRG_ER_FFAT_1': 'Interaction motif recognized by a major sperm protein domain.',
}


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fasta(path, sequences):
    Path(path).write_text(''.join(f'>{key}\n{sequence}\n' for key, sequence in sorted(sequences.items())), encoding='utf8')


def context(sequence, start, end):
    return sequence[max(0, start - 1 - 32): min(len(sequence), end + 32)]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = rows(INPUT / 'b1_samples.jsonl')
    parent_sequences = elm.read_fasta(elm.RAW / 'instances.fasta')
    external_parents, _, _ = elm.external_exclusions()
    for path in (OUT / 'raw_external').glob('batch*.fasta'):
        downloaded = elm.read_fasta(path)
        # ELM's archived parent sequence is authoritative for its coordinates.
        for accession, sequence in downloaded.items():
            parent_sequences.setdefault(accession, sequence)
    missing = sorted(external_parents - parent_sequences.keys())
    qparents = {r['parent_accession']: parent_sequences[r['parent_accession']] for r in samples}
    tparents = dict(qparents)
    tparents.update({p: parent_sequences[p] for p in external_parents if p in parent_sequences})
    qcontexts, context_info = {}, {}
    for r in samples:
        key = r['sample_id']
        seq = parent_sequences[r['parent_accession']]
        assert r['segment_sequence'] == seq[r['start']-1:r['end']]
        qcontexts[key] = context(seq, r['start'], r['end'])
        context_info[key] = {'kind': 'B1', 'sample_id': key, 'parent': r['parent_accession'], 'start': r['start'], 'end': r['end']}
    tcontexts = dict(qcontexts)
    external_context_audit = []
    counter = 0
    for name in ['benz_samples.jsonl', 't3_d2_pairs.jsonl', 't3_g1_records.jsonl']:
        for row in rows(ROOT / 'data/processed' / name):
            parents = {p['accession'] for p in row.get('parents', [])}
            parents.update(row.get('parent_accessions', []))
            if row.get('parent_accession'):
                parents.add(row['parent_accession'])
            segments = {row[k] for k in ['sequence', 'sequence_a', 'sequence_b'] if row.get(k)}
            for accession in sorted(parents):
                seq = parent_sequences.get(accession.split('-')[0])
                matches = []
                for segment in sorted(segments):
                    # Canonical exact occurrence only: mutations/synthetic additions
                    # are not silently stripped or invented to manufacture a match.
                    at = seq.find(segment) if seq else -1
                    if at >= 0:
                        matches.append((at + 1, at + len(segment), segment))
                external_context_audit.append({'source_file': name, 'source_id': row.get('sample_id', row.get('pair_id')),
                                               'parent': accession, 'exact_canonical_context_matches': len(matches),
                                               'context_resolution': 'verified_exact_parent_match' if matches else 'unresolved_variant_or_sequence'})
                for start, end, segment in matches:
                    key = f'external_context_{counter:06d}'
                    counter += 1
                    tcontexts[key] = context(seq, start, end)
                    context_info[key] = {'kind': 'external', 'parent': accession, 'start': start, 'end': end,
                                         'source_file': name, 'source_id': row.get('sample_id', row.get('pair_id'))}
    fasta(OUT / 'parent_queries.fasta', qparents)
    fasta(OUT / 'parent_targets.fasta', tparents)
    fasta(OUT / 'context_queries.fasta', qcontexts)
    fasta(OUT / 'context_targets.fasta', tcontexts)
    elm.write_json(OUT / 'context_mapping.json', context_info)
    elm.write_jsonl(OUT / 'external_context_audit.jsonl', external_context_audit)
    elm.write_json(OUT / 'alignment_protocol.json', PARAMETERS)
    meta = {'external_parents_requested': len(external_parents), 'external_parents_resolved': len(external_parents)-len(missing),
            'external_parents_missing': missing, 'b1_parent_queries': len(qparents), 'parent_targets': len(tparents),
            'b1_context_queries': len(qcontexts), 'context_targets': len(tcontexts),
            'external_unresolved_context_records': sum(not r['exact_canonical_context_matches'] for r in external_context_audit)}
    elm.write_json(OUT / 'sequence_manifest.json', meta)
    return meta


def run(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    with (OUT / 'mmseqs_execution.log').open('a', encoding='utf8') as handle:
        handle.write(json.dumps(command) + '\n' + result.stdout + '\n' + result.stderr + '\n')
    if result.returncode:
        raise RuntimeError(f'MMseqs failed: {command}; see mmseqs_execution.log')
    return result.stdout


def align():
    if not shutil.which('mmseqs'):
        raise RuntimeError('Use existing WSL mmseqs; no dependency is installed by this script')
    version = run(['mmseqs', 'version']).strip()
    for kind in ['parent', 'context']:
        work = OUT / ('mmseqs_' + kind)
        work.mkdir(exist_ok=True)
        qdb, tdb, pref, aln = [work / x for x in ['queries', 'targets', 'allpairs', 'alignments']]
        run(['mmseqs', 'createdb', str(OUT / f'{kind}_queries.fasta'), str(qdb), '--dbtype', '1'])
        run(['mmseqs', 'createdb', str(OUT / f'{kind}_targets.fasta'), str(tdb), '--dbtype', '1'])
        # Official fake-prefilter route: every query points to every target.
        target_index = Path(str(tdb) + '.index').read_bytes()
        pref.write_bytes(target_index)
        query_keys = [line.split('\t')[0] for line in Path(str(qdb) + '.index').read_text().splitlines()]
        Path(str(pref) + '.index').write_text(''.join(f'{key}\t0\t{len(target_index)}\n' for key in query_keys))
        Path(str(pref) + '.dbtype').write_bytes(bytes([7, 0, 0, 0]))
        p = PARAMETERS[kind]
        run(['mmseqs', 'align', str(qdb), str(tdb), str(pref), str(aln), '--alignment-mode', '3',
             '--min-seq-id', str(p['identity']), '-c', '0.8', '--cov-mode', '0', '-e', '1e-5',
             '--threads', '12', '--max-accept', '2147483647', '--max-rejected', '2147483647'])
        run(['mmseqs', 'convertalis', str(qdb), str(tdb), str(aln), str(OUT / f'{kind}_alignments.tsv'),
             '--format-output', 'query,target,fident,qcov,tcov,alnlen,evalue,qstart,qend,tstart,tend,qlen,tlen'])
    elm.write_json(OUT / 'tool_manifest.json', {'mmseqs_version': version, 'mode': 'exhaustive_all_query_target_pairs_no_kmer_prefilter',
        'official_recipe': 'https://github.com/soedinglab/MMseqs2/wiki#how-to-create-a-fake-prefiltering-for-all-vs-all-alignments'})


class UnionFind:
    def __init__(self, keys):
        self.parents = {key: key for key in keys}
    def find(self, key):
        self.parents.setdefault(key, key)
        if self.parents[key] != key:
            self.parents[key] = self.find(self.parents[key])
        return self.parents[key]
    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def accepted_alignment(fields, kind):
    p = PARAMETERS[kind]
    return (float(fields[2]) >= p['identity'] and float(fields[3]) >= p['query_coverage']
            and float(fields[4]) >= p['target_coverage'] and float(fields[6]) <= p['evalue']
            and (kind != 'context' or int(fields[5]) >= p['minimum_alignment_length']))


def build():
    samples = copy.deepcopy(rows(INPUT / 'b1_samples.jsonl'))
    external = elm.external_exclusions()[0]
    mappings = json.loads((OUT / 'context_mapping.json').read_text(encoding='utf8'))
    sequence_meta = json.loads((OUT / 'sequence_manifest.json').read_text(encoding='utf8'))
    parent_graph = UnionFind([r['parent_accession'] for r in samples] + list(external))
    family_graph = UnionFind(mappings)
    edges = []
    for kind, graph in [('parent', parent_graph), ('context', family_graph)]:
        for line in (OUT / f'{kind}_alignments.tsv').read_text(encoding='utf8').splitlines():
            fields = line.split('\t')
            if accepted_alignment(fields, kind):
                graph.union(fields[0], fields[1])
                if fields[0] != fields[1]:
                    edges.append({'kind': kind, 'query': fields[0], 'target': fields[1], 'identity': float(fields[2]),
                                  'qcov': float(fields[3]), 'tcov': float(fields[4]), 'alnlen': int(fields[5]),
                                  'evalue': float(fields[6])})
    by_sequence = {}
    for r in samples:
        previous = by_sequence.setdefault(r['segment_sequence'], r['sample_id'])
        family_graph.union(previous, r['sample_id'])
    partitions = {kind: defaultdict(set) for kind in ['parent', 'context']}
    for r in samples:
        partitions['parent'][parent_graph.find(r['parent_accession'])].add(r['split'])
        partitions['context'][family_graph.find(r['sample_id'])].add(r['split'])
    external_components = {'parent': {parent_graph.find(p) for p in external},
                           'context': {family_graph.find(k) for k, v in mappings.items() if v['kind'] == 'external'}}
    component_members = {kind: defaultdict(list) for kind in ['parent', 'context']}
    for kind, graph in [('parent', parent_graph), ('context', family_graph)]:
        for key in list(graph.parents):
            component_members[kind][graph.find(key)].append(key)
    component_ids = {kind: {key: kind + '_' + hashlib.sha256('|'.join(sorted(values)).encode()).hexdigest()[:16]
                           for key, values in groups.items()} for kind, groups in component_members.items()}
    audit = []
    for r in samples:
        reasons = []
        if not r['groups'].get('publication'):
            reasons.append('unresolved_primary_publication_source')
        for kind, key in [('parent', parent_graph.find(r['parent_accession'])), ('context', family_graph.find(r['sample_id']))]:
            if len(partitions[kind][key]) > 1:
                reasons.append(kind + '_homology_component_cross_partition')
            if key in external_components[kind]:
                reasons.append(kind + '_homology_component_contains_resolved_external')
            r['groups']['parent_homology' if kind == 'parent' else 'peptide_family'] = component_ids[kind][key]
        r['groups']['source_publication'] = list(r['groups']['publication'])
        r['homology_exclusion_reasons'] = reasons
        audit.append(r)
    grouped = defaultdict(list)
    for r in audit:
        if not r['homology_exclusion_reasons']:
            grouped[r['class_id']].append(r)
    classes = copy.deepcopy(rows(INPUT / 'b1_classes_main.jsonl'))
    class_audit = []
    eligible = set()
    for c in classes:
        retained = grouped[c['class_id']]
        independent_families = len({r['groups']['parent_homology'] for r in retained})
        item = {'class_id': c['class_id'], 'split': c['split'], 'post_component_records': len(retained),
                'parent_overlap_region_count': elm.independent_regions(retained),
                'independent_parent_homology_families': independent_families,
                'independence_count_rule': 'at most one independent unit per full-parent homology component per class',
                'minimum_independent_units': 10, 'eligible': independent_families >= 10}
        class_audit.append(item)
        if item['eligible']:
            eligible.add(c['class_id'])
    if eligible - REVIEWED_DESCRIPTIONS.keys():
        raise ValueError('Unreviewed class descriptions: ' + str(sorted(eligible - REVIEWED_DESCRIPTIONS.keys())))
    source_classes = {c['ELMIdentifier']: c for c in elm.read_tsv(elm.RAW / 'classes.tsv')[0]}
    description_audit = []
    candidate_lookup = {}
    for c in classes:
        c['eligible'] = c['class_id'] in eligible
        if not c['eligible']:
            continue
        original = source_classes[c['class_id']]
        reviewed = REVIEWED_DESCRIPTIONS[c['class_id']]
        description_audit.append({'class_id': c['class_id'], 'candidate_id': c['candidate_id'],
                                  'source_description_as_downloaded': original['Description'],
                                  'source_regex_withheld_from_model': original['Regex'],
                                  'reviewed_description': reviewed, 'review_status': 'reviewed_by_agent_against_original_ELM',
                                  'review_criteria': ['explicit_sequence_pattern_removed', 'residue_position_rules_removed',
                                                      'current_instance_answer_absent', 'original_function_and_partner_scope_preserved'],
                                  'source_ref': 'http://elm.eu.org/elms/' + c['class_id']})
        c['class_description_no_rule'] = reviewed
        c['description_review_required'] = False
        candidate_lookup[c['class_id']] = c['candidate_id']
    final = []
    for r in audit:
        if r['class_id'] not in eligible:
            r['homology_exclusion_reasons'].append('class_below_10_independent_parent_homology_families')
        if not r['homology_exclusion_reasons']:
            r['class_description_no_rule'] = REVIEWED_DESCRIPTIONS[r['class_id']]
            r['eligibility'].update({'parent_homology_verified': True, 'peptide_family_verified': True,
                                    'description_rule_reviewed': True})
            final.append(r)
    sequence_positives = defaultdict(set)
    for r in final:
        sequence_positives[r['segment_sequence']].add(r['class_id'])
    for r in final:
        r['known_positive_class_ids'] = sorted(sequence_positives[r['segment_sequence']])
        r['known_positive_candidate_ids'] = [candidate_lookup[key] for key in r['known_positive_class_ids']]
    nonempty = all(any(r['split'] == split for r in final) for split in ['train', 'validation', 'test'])
    fields = ['class_id', 'parent_homology', 'peptide_family', 'publication', 'segment_sequence']
    checks = {}
    for field in fields:
        owners = defaultdict(set)
        for r in final:
            value = r[field] if field in {'class_id', 'segment_sequence'} else r['groups'][field]
            for member in value if isinstance(value, list) else [value]:
                owners[member].add(r['split'])
        overlaps = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
        if overlaps:
            raise AssertionError(f'Actual retained partition leakage for {field}: {overlaps}')
        checks[field] = {'overlapping_groups': 0, 'verified': True, 'group_count': len(owners)}
    external_edges_retained = sum(1 for r in final if parent_graph.find(r['parent_accession']) in external_components['parent']
                                  or family_graph.find(r['sample_id']) in external_components['context'])
    assert external_edges_retained == 0
    group_audit = {'thresholds': PARAMETERS, 'partition_disjointness': checks, 'external_retained_component_edges': external_edges_retained,
                   'accepted_nonself_parent_edges': sum(e['kind'] == 'parent' for e in edges),
                   'accepted_nonself_context_edges': sum(e['kind'] == 'context' for e in edges),
                   'parent_components': [{'group_id': component_ids['parent'][key], 'members': sorted(values)}
                                         for key, values in component_members['parent'].items()],
                   'peptide_family_components': [{'group_id': component_ids['context'][key], 'members': sorted(values)}
                                               for key, values in component_members['context'].items()]}
    elm.write_jsonl(OUT / 'b1_samples.jsonl', final)
    elm.write_jsonl(OUT / 'test_gold.jsonl', [r for r in final if r['split'] == 'test'])
    elm.write_jsonl(OUT / 'b1_classes_main.jsonl', [c for c in classes if c['eligible']])
    elm.write_jsonl(OUT / 'b1_candidates_main.jsonl', [{'candidate_id': c['candidate_id'], 'description': c['class_description_no_rule']}
                                                    for c in classes if c['eligible']])
    elm.write_jsonl(OUT / 'b1_model_inputs.jsonl', [{'sample_id': r['sample_id'], 'panel_id': 'B1', 'task': 'unseen_function',
                    'molecule_scope': r['molecule_scope'], 'segment_sequence': r['segment_sequence'],
                    'candidate_catalog': 'b1_candidates_main.jsonl'} for r in final])
    elm.write_jsonl(OUT / 'class_independence_audit.jsonl', class_audit)
    elm.write_jsonl(OUT / 'description_review.jsonl', description_audit)
    elm.write_jsonl(OUT / 'exclusion_audit.jsonl', audit)
    elm.write_jsonl(OUT / 'accepted_homology_edges.jsonl', edges)
    elm.write_json(OUT / 'group_audit.json', group_audit)
    meta = {'panel_id': 'B1', 'scope': 'model_eukaryotes_homology_locked', 'records': len(final),
            'eligible_classes': len(eligible), 'unique_sequences': len(sequence_positives),
            'split_records': dict(Counter(r['split'] for r in final)),
            'split_classes': dict(Counter(c['split'] for c in classes if c['eligible'])),
            'held_out_test_class_ids': sorted({r['class_id'] for r in final if r['split'] == 'test'}),
            'independence_rule': '>=10 distinct full-parent homology components per class; region count reported separately',
            'all_partitions_nonempty': nonempty, 'external_sequence_coverage': sequence_meta,
            'class_seed_unchanged': True, 'input_manifest_sha256': digest(INPUT / 'b1_manifest.json'),
            'exclusion_counts': dict(Counter(reason for r in audit for reason in r['homology_exclusion_reasons']))}
    elm.write_json(OUT / 'b1_manifest.json', meta)
    limitations = ['Only two unseen test classes survive the conservative independence gate; report class-limited results.',
                   'Seven external parent sequences and 41 external variant/sequence context mappings are unresolved; isolation is restricted to resolved references.',
                   'Homology is an operational sequence threshold, not a guarantee against every remote evolutionary relationship.',
                   'Descriptions were reviewed by the coding agent against downloaded ELM records; no human review is claimed.',
                   'This lock supports panel-specific predictor evaluation, not a unified multitask connector claim.']
    pinned = [p for p in OUT.iterdir() if p.is_file() and p.name not in {'benchmark_lock.json', 'mmseqs_execution.log'}]
    pinned += [Path(__file__), elm.RAW / 'classes.tsv', elm.RAW / 'instances.tsv', elm.RAW / 'instances.fasta', INPUT / 'b1_samples.jsonl', INPUT / 'b1_manifest.json']
    relative = lambda path: path.relative_to(ROOT).as_posix()
    external_verified = not sequence_meta['external_parents_missing'] and not sequence_meta['external_unresolved_context_records']
    lock = {'scope': 'panel_specific_predictor', 'panel_id': 'B1',
            'status': ('frozen' if external_verified else 'frozen_restricted_scope') if nonempty else 'failed_empty_partition',
            'checks': {'peptide_family_disjoint': True, 'parent_homology_disjoint': True, 'source_disjoint': True,
                       'class_disjoint': True, 'description_rule_reviewed': True, 'external_isolation_verified': external_verified},
            'limitations': limitations, 'pins': {relative(path): digest(path) for path in pinned},
            'homology_evidence': {'protocol_path': relative(OUT / 'alignment_protocol.json'),
                                 'alignment_paths': [relative(OUT / f'{kind}_alignments.tsv') for kind in ['parent', 'context']],
                                 'group_audit_path': relative(OUT / 'group_audit.json')}}
    elm.write_json(OUT / 'benchmark_lock.json', lock)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'align', 'build', 'all'], default='all')
    args = parser.parse_args()
    if args.stage in {'prepare', 'all'}:
        print(json.dumps(prepare(), indent=2), flush=True)
    if args.stage in {'align', 'all'}:
        align()
    if args.stage in {'build', 'all'}:
        print(json.dumps(build(), indent=2))
