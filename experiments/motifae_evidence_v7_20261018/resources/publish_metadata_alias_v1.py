"""Additional publication-only entrypoint; frozen evaluation files stay intact.

An aggregate comparator named 'sequence' collided with the raw-sequence privacy
guard. Rename ONLY three typed aggregate-control keys in the public derivative.
The original guard remains enabled for every field and every other file.
"""
import argparse,hashlib,importlib.util,json,math
from datetime import datetime,timezone
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def alias(value):
    value=json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
    comparison=value.get('comparison',{})
    for name in ('qwen_activity_overclaim_rates','qwen_class_recovery_rates'):
        if name not in comparison:continue
        mapping=comparison[name]
        if 'sequence' not in mapping:raise ValueError('Expected aggregate control key missing')
        number=mapping['sequence']
        if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or not 0<=number<=1:
            raise ValueError('Only scalar aggregate rates may be aliased')
        if 'sequence_control' in mapping:raise ValueError('Alias collision')
        mapping['sequence_control']=mapping.pop('sequence')
    arms=value.get('qa',{}).get('arms',{})
    if 'sequence' in arms:
        envelope=arms['sequence']
        if set(envelope)!={'class','position','activity','evidence'} or not all(isinstance(v,dict) for v in envelope.values()):
            raise ValueError('Only exact question aggregate envelope may be aliased')
        if 'sequence_control' in arms:raise ValueError('Alias collision')
        arms['sequence_control']=arms.pop('sequence')
    value['publication_control_aliases']={'sequence_control':'sequence (raw-sequence input comparator)'}
    return value

def publish(scope):
    scope=Path(scope).resolve();manifest=scope/'frozen_manifest.json'
    closure=json.loads(manifest.read_text(encoding='utf-8'))
    for name,expected in closure['source_pins'].items():
        if sha(scope/name)!=expected:raise ValueError('Frozen scientific source changed')
    source=scope/'reports/SUMMARY.json';original_sha=sha(source)
    if json.loads(source.read_text(encoding='utf-8'))['status']!='complete_pilot_no_incremental_gain':
        raise ValueError('This recovery is restricted to the completed observed pilot')
    spec=importlib.util.spec_from_file_location('frozen_v7_publisher',scope/'resources/publish_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Positive aggregate fixture, then negative raw-sequence fixtures: the guard
    # must still reject real sequence payloads, including nested ones.
    module.private_guard(alias({'comparison':{'qwen_class_recovery_rates':{'sequence':.5}}}))
    for raw in ({'sequence':'ACDEFGHIKLMNPQRSTVWY'}, {'comparison':{'other':{'sequence':'ACDEFGHIKLMNPQRSTVWY'}}}):
        try:module.private_guard(alias(raw))
        except ValueError:pass
        else:raise AssertionError('Private sequence guard weakened')
    checked=module.checked_public_bytes
    def public_bytes(path,replacements):
        if path.resolve()!=source.resolve():return checked(path,replacements)
        value=alias(json.loads(path.read_text(encoding='utf-8')))
        value=module.sanitize(value,replacements);module.private_guard(value)
        data=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
        return data,'publication_only_aggregate_control_alias_and_runtime_paths'
    module.checked_public_bytes=public_bytes
    result=module.publish(scope,'final')
    if sha(source)!=original_sha or sha(manifest)!=sha(scope/'frozen_manifest.json'):
        raise ValueError('Original result unexpectedly changed')
    receipt={'schema':'motifae-publication-metadata-alias.v1','status':'published_verified',
        'at_utc':datetime.now(timezone.utc).isoformat(),'commit':result['commit'],
        'source_summary_sha256':original_sha,'frozen_manifest_sha256':sha(manifest),
        'additional_operational_entrypoint_sha256':sha(Path(__file__)),
        'frozen_publisher_source_sha256':sha(scope/'resources/publish_pilot.py'),
        'private_guard_retained':True,'privacy_selftests_passed':3,
        'scientific_inputs_metrics_and_decision_unchanged':True,'verified':result['verified']}
    module.write(scope/'private/publication_metadata_alias_v1_receipt.json',receipt)
    status=module.read(scope/'STATUS.json')
    status['stages']['publication_recovery']={'status':'complete','commit':result['commit'],'finished_at':receipt['at_utc'],
        'receipt_sha256':sha(scope/'private/publication_metadata_alias_v1_receipt.json')}
    status['status']='complete_publication_recovered';status['updated_at']=receipt['at_utc']
    module.write(scope/'STATUS.json',status)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);a=p.parse_args();publish(a.scope)
