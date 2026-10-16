"""Target/context/state matched donors, never an unmatched fallback."""
import hashlib
import json

from .contracts import ContractError

def _stable(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def _identity(row):
    panel=row.get('panel_id')
    reasons=[]
    group=row.get('independent_group_id')
    if not group:
        reasons.append('independent_family_or_experiment_group_missing')
    if row.get('split') not in ('test','external'):
        reasons.append('receiver_not_in_locked_test_or_external_split')
    sequence=row.get('peptide_sequence',row.get('segment_sequence'))
    if not sequence:
        reasons.append('molecular_sequence_missing')
    if panel=='A1':
        target=row.get('target_sequence')
        context=row.get('experimental_context',{}).get('comparable_condition_group')
        state=row.get('chemical_state')
        if not target:
            reasons.append('target_sequence_missing')
        if not context:
            reasons.append('experimental_condition_comparability_unverified')
        if not isinstance(state,dict) or state.get('explicitly_representable') is not True:
            reasons.append('chemical_state_unverified')
    elif panel=='A2':
        target=None
        context=row.get('condition_id')
        state=row.get('modifications')
        if not context or state is None:
            reasons.append('RT_condition_or_chemical_state_missing')
    else:
        target=row.get('target_sequence')
        context=row.get('question_id',row.get('task'))
        state=row.get('chemical_state',row.get('molecule_scope'))
        if not context or state is None:
            reasons.append('task_or_molecular_state_missing')
    background=row.get('allowed_background_id',row.get('allowed_background',''))
    key=(panel,row.get('task'),row.get('split'),target,
         row.get('target_construct_id',row.get('target_sequence_role')),
         _stable(context),_stable(state),_stable(background))
    return key,group,len(sequence or ''),reasons

def matched_donors(records,n=10,seed=42,length_tolerance=2):
    if n<1 or length_tolerance<0:
        raise ContractError('positive donor count and nonnegative length tolerance required')
    ids=[row.get('sample_id') for row in records]
    if any(not x for x in ids) or len(ids)!=len(set(ids)):
        raise ContractError('unique sample IDs are required')
    metadata=[_identity(row) for row in records]
    result=[]
    for i,row in enumerate(records):
        key,group,length,reasons=metadata[i]
        candidates=[]
        if not reasons:
            for j,other in enumerate(records):
                other_key,other_group,other_length,other_reasons=metadata[j]
                if i!=j and not other_reasons and key==other_key and group!=other_group and abs(length-other_length)<=length_tolerance:
                    candidates.append(other['sample_id'])
        candidates.sort(key=lambda donor:hashlib.sha256(f'{seed}|{row["sample_id"]}|{donor}'.encode()).digest())
        result.append({'sample_id':row['sample_id'],'donor_ids':candidates[:n],
                       'eligible_donors':len(candidates),'requested_donors':n,
                       'coverage_complete':len(candidates)>=n,'unmatchable_reasons':reasons})
    return result
