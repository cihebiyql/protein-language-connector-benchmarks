"""Freeze a new same-peptide panel; never mutate earlier benchmark splits."""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import urllib.request

SOURCE_COMMIT = '5187be8348ecea98565ef56c7be51a8dc49d8524'
SOURCE_SHA = 'cd4ab4b21c5a2855712e728a402ca4442795d617da82cf7a7a6db4b4f1a12917'
SOURCE_URL = ('https://raw.githubusercontent.com/MannLabs/DeepCollisionalCrossSection/'
              + SOURCE_COMMIT + '/output/evidence_aligned_PT.csv')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def median(values):
    values = sorted(values); n = len(values)
    return values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2

def prepare(scope, old_scope):
    scope, old_scope = Path(scope), Path(old_scope)
    root = scope/'data'
    if (root/'manifest.json').exists():
        raise FileExistsError('A frozen panel already exists; do not regenerate after outcomes')
    source = scope/'resources/evidence_aligned_PT.csv'
    if not source.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
            source.write_bytes(response.read())
    if sha(source) != SOURCE_SHA:
        raise ValueError('Paired measured source differs from fixed author version')
    old_sequences = set(); old_pins = {}
    for task in ('meltome','elm','rt'):
        for file in sorted((old_scope/'data'/task).glob('*.jsonl')):
            if file.name == 'candidates.jsonl': continue
            old_pins[str(file.relative_to(old_scope))] = sha(file)
            old_sequences.update(r['sequence'] for r in read_rows(file))
    exclusions = collections.Counter(); candidates = {}
    with source.open(encoding='utf-8', newline='') as stream:
        for raw in csv.DictReader(stream):
            sequence = raw['Sequence']
            if not re.fullmatch('[ADEFGHIKLMNPQRSTVWY]{8,30}', sequence):
                exclusions['length_noncanonical_or_cysteine'] += 1; continue
            if raw['Modified sequence'].strip('_') != sequence or int(raw['Charge']) != 2:
                exclusions['modified_or_not_charge2'] += 1; continue
            if sequence in old_sequences:
                exclusions['prior_v5_sequence_quarantine'] += 1; continue
            rt, ccs = float(raw['Retention time']), float(raw['CCS'])
            if not all(math.isfinite(v) and v>0 for v in (rt,ccs)):
                exclusions['nonpositive_or_nonfinite_label'] += 1; continue
            if sequence in candidates:
                raise ValueError('Duplicate chemical state must be resolved before freezing')
            candidates[sequence] = dict(id=digest(sequence), sequence=sequence,
                experiment=raw['Experiment'], rt=rt, ccs=ccs, charge=2)
    # Label-independent bounded panel selection, then connected-component grouping.
    sequences = sorted(candidates, key=lambda s:digest('panel-v1\0'+s))[:4096]
    if len(sequences) != 4096: raise ValueError('Insufficient paired experimental peptides')
    parent = list(range(len(sequences)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i,j):
        i,j = find(i),find(j)
        if i != j: parent[max(i,j)] = min(i,j)
    owners = {}
    for i,sequence in enumerate(sequences):
        for kmer in {sequence[j:j+8] for j in range(len(sequence)-7)}:
            if kmer in owners: union(i,owners[kmer])
            else: owners[kmer] = i
    components = collections.defaultdict(list)
    for i,sequence in enumerate(sequences): components[find(i)].append(sequence)
    rows = []
    for members in components.values():
        group = digest('\n'.join(sorted(members)))
        bucket = int(digest('split-v1\0'+group)[:16],16)%100
        split = 'train' if bucket<70 else 'validation' if bucket<85 else 'test'
        for sequence in members:
            rows.append({**candidates[sequence], 'group_id':group, 'split':split})
    rows.sort(key=lambda r:r['id'])
    by_split = {s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    if len(by_split['train'])<2000 or min(len(by_split[s]) for s in ('validation','test'))<400:
        raise ValueError('Fixed component split lacks sufficient support; no silent resplitting')
    thresholds = {k:median([r[k] for r in by_split['train']]) for k in ('rt','ccs')}
    mappings = {}; rng = random.Random(991)
    # Same experiment and coarse length strata; peptide donors never depend on labels.
    for split, values in by_split.items():
        buckets = collections.defaultdict(list)
        for r in values: buckets[(r['experiment'],(len(r['sequence'])-8)//4)].append(r['id'])
        mapping = {}
        # Merge singleton length bins into the nearest occupied bin of the same
        # acquisition cohort, before any scores are computed or panel is frozen.
        for key in sorted(list(buckets)):
            if len(buckets[key]) != 1: continue
            neighbors = [k for k in buckets if k[0]==key[0] and k!=key and buckets[k]]
            if not neighbors: raise ValueError('Acquisition cohort has no control donor')
            target = min(neighbors,key=lambda k:(abs(k[1]-key[1]),k[1]))
            buckets[target].extend(buckets.pop(key))
        for ids in buckets.values():
            ids = sorted(ids); rng.shuffle(ids)
            mapping.update({v:ids[(i+1)%len(ids)] for i,v in enumerate(ids)})
        mappings[split] = mapping
        out = root/(split+'.jsonl'); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in values),encoding='utf-8')
    write(root/'control_mapping.json', mappings)
    manifest = {'schema':'same-peptide-paired-panel.v1','source_url':SOURCE_URL,
        'source_sha256':SOURCE_SHA,'source_commit':SOURCE_COMMIT,
        'label_semantics':{'rt':'author selected highest-intensity experimental PT feature retention time, minutes',
            'ccs':'author run-aligned average measured TIMS CCS, angstrom_squared'},
        'acquisition_population':'ProteomeTools synthetic peptides, uniform paper LC method; three Experiment cohorts',
        'not_original_paper_split':True,'raw_instrument_alignment_precomputed_by_authors':True,
        'filters':'charge2;8-30 canonical residues;noC;no variable modifications;positivefinite labels',
        'excluded':dict(exclusions),'eligible_after_quarantine':len(candidates),'panel_sequences':len(rows),
        'panel_selection':'first4096 SHA256(panel-v1 NUL sequence); independent of label values and outcomes',
        'grouping':'connected components sharing any exact8mer among panel; no claim of complete homology exclusion',
        'groups':len(components),'largest_group':max(map(len,components.values())),
        'split_rule':'SHA256(split-v1 NUL group)%100:70/15/15; no QA-row split',
        'counts':{s:len(v) for s,v in by_split.items()},'thresholds':thresholds,'threshold_source':'TRAIN medians only',
        'old_v5_sequences_quarantined':len(old_sequences),'old_v5_frozen_file_pins':old_pins,
        'question_types':['rt','ccs','both'],'independent_experimental_endpoints':2,
        'shuffled_control':'split-local acquisition cohort and four-residue length bins; singleton bins merged to nearest same-cohort bin; frozen RNG991 cyclic derangement',
        'both_is_composition_not_third_independent_property':True,
        'input_exclusions':['id','group_id','raw_sequence','measured Mass','Intensity','Score','measured RT','measured CCS'],
        'safe_conditions':['charge2','experiment','common acquisition method'],
        'files':{p.name:sha(p) for p in sorted(root.glob('*.json*'))}}
    write(root/'manifest.json',manifest)
    return manifest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--old-scope',required=True)
    a=p.parse_args();m=prepare(a.scope,a.old_scope)
    print(json.dumps({k:m[k] for k in ('counts','thresholds','groups','largest_group','eligible_after_quarantine')},indent=2))
