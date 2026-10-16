"""Join every published quantitative row to completed per-seed predictions."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/real_run_20261015'
SEEDS=(17,42,2026)
METHODS=('MLP','L30-J0','L15-J14')

def main():
    source=ROOT/'data/processed/t3_g1_records.jsonl'
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==29
    predictions={}
    fingerprints={'data/processed/t3_g1_records.jsonl':hashlib.sha256(source.read_bytes()).hexdigest()}
    for method in METHODS:
        predictions[method]={}
        for seed in SEEDS:
            path=(OUT/f't3_mlp_seed{seed}.scores.npz' if method=='MLP' else
                  ROOT/f'results/real_t3_20261015/{method}_seed{seed}/test_scores.npz')
            arrays=np.load(path,allow_pickle=False)
            pair_ids=arrays['g1_direction__pair_ids'].astype(str).tolist()
            assert len(pair_ids)==17 and len(set(pair_ids))==17
            for mode in ('correct','off','swapped_messages'):
                key_mode='closed_raw_zero' if method=='MLP' and mode=='off' else mode
                scores=arrays[f'g1_direction__{key_mode}']
                assert scores.shape==(17,2) and np.isfinite(scores).all()
                for pair_id,label in zip(pair_ids,np.argmax(scores,axis=1)):
                    predictions[method].setdefault(pair_id,{}).setdefault(mode,[]).append('AB'[label])
            fingerprints[str(path.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(path.read_bytes()).hexdigest()
    records=[]
    for row in rows:
        record={key:row.get(key,'') for key in (
            'pair_id','source_table_row','protein_domain','gene_mutation_as_published','gene',
            'sequence_a','sequence_b','kd_wt_uM_as_published','kd_mut_uM_as_published',
            'label','direction_status','sequence_source_consistent','group_id','series_id')}
        record['identity_notes']='; '.join(row.get('identity_notes',[]))
        for method in METHODS:
            candidate=predictions[method].get(row['pair_id'])
            for mode in ('correct','off','swapped_messages'):
                votes=candidate[mode] if candidate else []
                record[f'{method}_{mode}_seed17_42_2026']='|'.join(votes)
                record[f'{method}_{mode}_majority']=max('AB',key=votes.count) if votes else ''
            record[f'{method}_correct_seed_fraction']=(sum(x==row['label'] for x in candidate['correct'])/3 if candidate else '')
        records.append(record)
    path=OUT/'g1_all_quantitative_rows.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT/'g1_case_sources.json').write_text(json.dumps({
        'rows':29,'scored_rows':17,'seed_order':list(SEEDS),
        'majority_vote_scope':'display only; primary scores average the individual seed metrics',
        'sha256':fingerprints,'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    # These genes were selected before inspecting model outcomes, as published quantitative examples.
    selected=[row for row in records if row['gene'] in ('CDC45','SQSTM1','CREBBP') and row['direction_status']=='direction_resolvable']
    print(json.dumps(selected,ensure_ascii=False))

if __name__=='__main__':
    main()
