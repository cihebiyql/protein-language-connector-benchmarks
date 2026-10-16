"""Bounded B1 -> frozen Qwen execution check; ranking is inherited from B1."""
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

def main():
    torch.manual_seed(17)
    torch.set_num_threads(8)
    path = ROOT/'models/Qwen--Qwen2.5-1.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    predictions_path = ROOT/'results/b1_test_predictions.jsonl'
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines()]
    examples = []
    for prediction in predictions[:3]:
        top3 = prediction['ranked_target_ids'][:3]
        # Do not include observed targets, sequence identity, or source annotations in the prompt.
        prompt = ('A frozen sequence classifier ranked these candidate targets: ' + json.dumps(top3)
                  + '. Return JSON only with keys ranked_target_ids (copy these three IDs in order) and interpretation '
                  + '(one short English sentence stating these are predicted candidate interactions and no cellular consequence '
                  + 'is established). Do not add mechanistic or experimental claims.')
        tokens = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True,
                                               add_generation_prompt=True, return_tensors='pt').cuda()
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(tokens, attention_mask=torch.ones_like(tokens), max_new_tokens=128,
                                       do_sample=False, temperature=None, top_p=None, top_k=None,
                                       pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(generated[0,tokens.shape[1]:], skip_special_tokens=True)
        cleaned = response.strip()
        if cleaned.startswith('```json') and cleaned.endswith('```'):
            cleaned = cleaned[7:-3].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = None
        examples.append({'sample_id':prediction['sample_id'], 'input_ranked_target_ids':top3,
                         'response':response, 'json_parse_success':isinstance(parsed,dict),
                         'ranking_preserved':isinstance(parsed,dict) and parsed.get('ranked_target_ids')==top3,
                         'generation_seconds':time.perf_counter()-started})
    result = {'scope':'three_fixed_test_examples_pipeline_canary_only_not_explanation_accuracy',
              'ranking_source':'results/b1_test_predictions.jsonl; no reranking by Qwen',
              'predictor_predictions_sha256':hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
              'sample_selection':'first three rows in fixed test manifest order, no outcome selection',
              'decode':'greedy, max_new_tokens=128', 'examples':examples,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'results/b2_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
