"""Export allowed model inputs separately from gold labels and source metadata."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bench_v02.contracts import ROOT,ContractError,jsonl,record_reasons,sha256

def model_input(record,panel):
    reasons=record_reasons(record,panel)
    if reasons:
        raise ContractError(f'ineligible model input: {reasons}')
    if panel=='A1':
        allowed={'peptide_sequence':record['peptide_sequence'],'chemical_state':record['chemical_state'],
                 'target_sequence':record['target_sequence'],'target_sequence_role':record['target_sequence_role'],
                 'experimental_context':{k:record.get('experimental_context',{}).get(k) for k in
                    ('assay','temperature_kelvin','ph','buffer','actual_construct_sequence','actual_construct_verified')}}
        question='Predict pKD = -log10(KD in mol/L) from the provided molecular inputs and declared conditions. Return JSON with numeric pKD.'
    elif panel=='A2':
        allowed={'peptide_sequence':record['peptide_sequence'],'peptidoform':record['peptidoform'],
                 'modifications':record['modifications'],'condition_id':record['condition_id'],
                 'condition_metadata':record.get('rawmetadata',{}).get('condition_metadata',{})}
        question='Predict retention time in minutes under the declared chromatographic condition. Return JSON with numeric retention_time and unit min.'
    else:
        raise ContractError('B1 uses the separately sanitized anonymous semantic catalog export')
    return {'sample_id':record['sample_id'],'panel_id':panel,'task':record['task'],
            'model_inputs':allowed,'question':question,
            'transport_metadata_not_prompt_fields':['sample_id','panel_id','task']}

def build():
    out=ROOT/'data/processed/bench_v02'
    config=json.loads((ROOT/'configs/bench_v02.json').read_text(encoding='utf-8'))
    manifests=[]
    for panel,key in [('A1','source_records'),('A2','target_records')]:
        source=ROOT/config['panels'][panel][key]
        dest=out/f'{panel.lower()}_model_inputs.jsonl'
        count=0;long_targets=0
        with dest.open('w',encoding='utf-8') as handle:
            for row in jsonl(source):
                value=model_input(row,panel)
                handle.write(json.dumps(value,ensure_ascii=False)+'\n')
                count+=1
                long_targets+=len(row.get('target_sequence') or '')>1022
        manifests.append({'panel_id':panel,'source':str(source.relative_to(ROOT)),
                          'source_sha256':sha256(source),'model_inputs':str(dest.relative_to(ROOT)),
                          'model_inputs_sha256':sha256(dest),'rows':count,
                          'target_sequences_over_1022_residues':long_targets})
    result={'status':'allowed_input_views_only_not_training_authorization',
            'gold_or_measurement_values_in_model_inputs':False,
            'raw_metadata_or_source_DOI_in_model_inputs':False,'datasets':manifests}
    (out/'input_views_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    print(json.dumps(build(),ensure_ascii=False))
