"""Fail-closed scientific contracts; no model training is performed here."""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
AA=set('ACDEFGHIKLMNPQRSTVWY')
CHANNELS={'generation','readout_head','predictor','oracle_pipeline'}

class ContractError(ValueError):
    pass

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def json_generation(raw_text):
    """Parse an actual generated object; never mine labels or numbers from prose."""
    if not isinstance(raw_text,str) or not raw_text.strip():
        raise ContractError('generation requires raw generated text')
    text=raw_text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text=text[7:-3].strip()
    try:
        result=json.loads(text)
    except (ValueError,TypeError) as error:
        raise ContractError('generated answer is not a JSON object') from error
    if not isinstance(result,dict):
        raise ContractError('generated answer must be a JSON object')
    return result

def numeric_generation(raw_text,key,expected_unit=None):
    result=json_generation(raw_text)
    if not finite_number(result.get(key)):
        raise ContractError(f'generated {key} must be a finite numeric JSON field')
    if expected_unit is not None and result.get('unit')!=expected_unit:
        raise ContractError('generated unit does not match the locked endpoint')
    return float(result[key])

def record_reasons(record,panel):
    """Row eligibility is distinct from globally locked benchmark readiness."""
    reasons=[]
    if record.get('panel_id')!=panel or not record.get('sample_id'):
        reasons.append('missing_or_wrong_sample_panel_identity')
    sequence=record.get('peptide_sequence',record.get('segment_sequence'))
    if not sequence or not set(sequence)<=AA:
        reasons.append('peptide_sequence_missing_or_not_canonical')
    if panel=='A1':
        target=record.get('target_sequence')
        if not target or not set(target)<=AA:
            reasons.append('real_target_sequence_required_opaque_id_insufficient')
        if record.get('measurement_type')!='KD':
            reasons.append('endpoint_is_not_KD')
        if record.get('relation')!='=' or not record.get('label_exact'):
            reasons.append('censored_or_unverified_value_is_not_exact_regression')
        kd=record.get('kd_molar')
        if not finite_number(kd) or kd<=0 or not finite_number(record.get('pkd')):
            reasons.append('invalid_KD_or_pKD')
        elif abs(-math.log10(kd)-record['pkd'])>1e-8:
            reasons.append('pKD_unit_transform_mismatch')
        if sequence and not 5<=len(sequence)<=50:
            reasons.append('peptide_outside_primary_5_50_range')
        chemical=record.get('chemical_state',{})
        if not isinstance(chemical,dict) or str(chemical.get('topology','')).lower()!='linear' or chemical.get('explicitly_representable') is not True:
            reasons.append('linear_chemical_state_not_verified')
        if not record.get('eligibility',{}).get('broad_reference_target'):
            reasons.append('source_row_not_broad_eligible')
    elif panel=='A2':
        if not record.get('condition_id') or record.get('rt_unit')!='min' or not finite_number(record.get('rt_value')):
            reasons.append('RT_condition_scale_or_unit_unresolved')
        elif record['rt_value']<0:
            reasons.append('negative_raw_coordinate_is_not_verified_elapsed_minutes')
        if not isinstance(record.get('modifications'),list) or not record.get('peptidoform'):
            reasons.append('peptidoform_state_not_preserved')
        if not record.get('eligibility',{}).get('measurement_valid'):
            reasons.append('RT_measurement_not_verified')
    elif panel=='B1':
        if not record.get('class_id') or not record.get('class_description_no_rule'):
            reasons.append('class_semantics_missing')
        if record.get('molecule_scope')!='local_functional_region_in_parent_protein' or record.get('free_peptide_activity_claim'):
            reasons.append('local_region_scope_must_not_become_free_peptide_drug_claim')
        if not all(record.get('eligibility',{}).get(k) for k in ('sequence_verified','no_rule_prompt')):
            reasons.append('sequence_or_rule_redaction_unverified')
    return reasons

def validate_run_manifest(manifest,panel,*,diagnostic=False,protocol_path=None,gold_path=None,benchmark_lock_path=None):
    if manifest.get('benchmark_version')!='0.2':
        raise ContractError('legacy results cannot be relabeled as v0.2')
    channel=manifest.get('channel')
    if channel not in CHANNELS:
        raise ContractError('declare generation/readout_head/predictor/oracle_pipeline separately')
    if not isinstance(manifest.get('model_id'),str) or not manifest['model_id'].strip():
        raise ContractError('model_id is required')
    if type(manifest.get('seed')) is not int or manifest['seed'] not in (17,42,2026):
        raise ContractError('declare a registered training seed')
    if not isinstance(manifest.get('checkpoint_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',manifest['checkpoint_sha256']):
        raise ContractError('checkpoint SHA256 is required')
    if manifest.get('role') not in ('controlled_interface','controlled_predictor','published_system','diagnostic_oracle'):
        raise ContractError('declare controlled interface/predictor, published system, or diagnostic oracle role')
    if channel=='oracle_pipeline' and manifest.get('role')!='diagnostic_oracle':
        raise ContractError('oracle facts are a diagnostic upper bound, not a main competing model')
    if manifest.get('legacy_alias_logits'):
        raise ContractError('legacy alias logits are diagnostic v0.1, not v0.2 generation')
    if channel=='readout_head' and (manifest.get('state_boundary')!='pre_answer' or manifest.get('included_answer_tokens') is not False):
        raise ContractError('readout/probe state must be recorded before the correct answer')
    if manifest.get('role')=='controlled_interface':
        checkpoints=manifest.get('source_checkpoints_by_panel',{})
        if not all(checkpoints.get(p) for p in ('A1','A2','B1','B2')) or len(set(checkpoints.values()))!=1:
            raise ContractError('main interfaces require one source multitask checkpoint per seed')
        if not all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in checkpoints.values()):
            raise ContractError('source checkpoint identities must be SHA256 values')
        source=next(iter(checkpoints.values()))
        declared_source=(manifest.get('adaptation') or {}).get('parent_source_checkpoint_sha256',manifest['checkpoint_sha256'])
        if declared_source!=source:
            raise ContractError('run checkpoint does not match the common multitask source')
    if manifest.get('adaptation'):
        a=manifest['adaptation']
        ids=a.get('label_ids',{})
        used=set().union(*(set(ids.get(k,[])) for k in ('fit','calibration','validation','tuning')))
        if len(used)>a.get('label_budget',-1):
            raise ContractError('all fit/calibration/validation/tuning labels count toward budget')
        if used&set(a.get('test_ids',[])):
            raise ContractError('adaptation labels overlap the locked target test set')
        if not a.get('parent_source_checkpoint_sha256'):
            raise ContractError('adaptation must identify the common source checkpoint')
    if not diagnostic:
        if protocol_path is None or gold_path is None:
            raise ContractError('formal evaluation requires physical protocol and gold paths')
        lock_path=Path(benchmark_lock_path) if benchmark_lock_path else ROOT/'data/processed/bench_v02/benchmark_lock.json'
        if not lock_path.is_file():
            raise ContractError('phase0 external/class/homology lock file has not been created')
        if manifest.get('benchmark_lock_sha256')!=sha256(lock_path):
            raise ContractError('registered benchmark lock fingerprint mismatch')
        lock=json.loads(lock_path.read_text(encoding='utf-8'))
        if lock.get('status')!='frozen' or not all(lock.get(k) for k in
            ('external_experiments_excluded','homology_split_verified','class_holdout_verified')):
            raise ContractError('phase0 external/class/homology locks are incomplete')
        if panel not in lock.get('ready_panels',[]):
            raise ContractError('panel is not ready in the frozen benchmark lock')
        if protocol_path is not None and lock.get('protocol_sha256')!=sha256(protocol_path):
            raise ContractError('protocol fingerprint mismatch')
        if gold_path is not None and lock.get('gold_sha256')!=sha256(gold_path):
            raise ContractError('gold data fingerprint mismatch')
        checkpoint=manifest.get('checkpoint_path')
        if not checkpoint:
            raise ContractError('formal runs must locate the actual checkpoint file')
        checkpoint=Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint=ROOT/checkpoint
        if not checkpoint.is_file() or sha256(checkpoint)!=manifest['checkpoint_sha256']:
            raise ContractError('actual checkpoint file/fingerprint mismatch')
        legacy_receipt=ROOT/'results/real_run_20261015/final_verification.json'
        if legacy_receipt.exists():
            old=json.loads(legacy_receipt.read_text(encoding='utf-8'))
            legacy={v for k,v in old.get('source_sha256',{}).items() if k.endswith(('.checkpoint.pt','/best.pt'))}
            ancestry={manifest['checkpoint_sha256'],*manifest.get('source_checkpoints_by_panel',{}).values()}
            ancestry.add((manifest.get('adaptation') or {}).get('parent_source_checkpoint_sha256'))
            if ancestry&legacy:
                raise ContractError('legacy task-trained checkpoints cannot become fresh v0.2 external-zero-exposure results')
    return channel
