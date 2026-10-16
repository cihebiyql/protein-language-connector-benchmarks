"""Recompute archived predictions without models, training, downloads, or GPUs.

Run ``python -m analysis.analyze_results validate`` to check the public release,
or ``... analyze --output-dir <new-directory>`` to export auditable CSV/JSON.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

SCENARIOS = ("new_peptide_known_target", "new_peptide_new_target")
SEEDS = (17, 42, 2026)
COUNTS = {SCENARIOS[0]: {"train": 2062, "validation": 147, "test": 152},
          SCENARIOS[1]: {"train": 2063, "validation": 431, "test": 442}}
DATA_SHA256 = {
    SCENARIOS[0]: {"samples.jsonl": "d1d9ce44db4aeefcc829f5ce7b50b3e288ad03a28af9d15bb9f1fd550a6b657e",
                   "test_gold.jsonl": "6f50e85443d1056ef35d509b5229dfb0bc364e266da2aebfa3fa78dcdb07c07b"},
    SCENARIOS[1]: {"samples.jsonl": "7ff5a67203efaa94db29fad1f7d2774b5f5e5f8a9155fc47a090e0f2a3baee71",
                   "test_gold.jsonl": "388e50ac4776bd122dc3fb60a688ee715173a8703c890ac7d0ad38eaab82065f"}}
CONTRASTS = ("depth_real", "depth_neutral", "content_early", "content_mid",
             "depth_content_difference_in_differences", "gate_zero_initialization", "balanced_pool_allocation")
ATOL = 1e-12  # Arithmetic reconstruction tolerance, never a scientific superiority threshold.


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def no_duplicate_keys(pairs):
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate_key")
    return dict(pairs)


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def strict_loads(text):
    return json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant)


def parse_answer(text):
    """Original strict one-field parser, including its stored reason labels.

    Extreme integer/recursion inputs are rejected rather than crashing; actual
    archived answers are checked against their stored original parse records.
    """
    try:
        value = strict_loads(text)
    except (ValueError, TypeError, RecursionError, OverflowError):
        return {"valid": False, "value": None, "reason": "invalid_json_or_nonfinite"}
    if not isinstance(value, dict) or set(value) != {"pKD"}:
        return {"valid": False, "value": None, "reason": "wrong_schema"}
    if not finite(value["pKD"]):
        return {"valid": False, "value": None, "reason": "nonfinite_or_nonnumeric_pKD"}
    return {"valid": True, "value": float(value["pKD"]), "reason": None}


def average_ranks(values):
    """Stable average ranks, one-based, with exact-value ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def spearman(gold, predictions):
    if len(gold) != len(predictions) or any(not finite(v) for v in [*gold, *predictions]):
        raise ValueError("Spearman needs equal-length finite vectors")
    if len(gold) < 2:
        return None
    x, y = average_ranks(gold), average_ranks(predictions)
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    x, y = [v - mx for v in x], [v - my for v in y]
    denominator = math.sqrt(math.fsum(v*v for v in x)) * math.sqrt(math.fsum(v*v for v in y))
    return max(-1., min(1., math.fsum(a*b for a,b in zip(x,y)) / denominator)) if denominator else None


def regression(rows, values):
    if len(rows) != len(values) or any(not finite(x) for x in values):
        raise ValueError("Regression needs complete finite aligned predictions")
    gold = [r["pkd"] for r in rows]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["groups"]["target_sequence_sha256"]].append(i)
    per_group = []
    for group_id, positions in groups.items():
        errors = [abs(gold[i] - values[i]) for i in positions]
        per_group.append({"group_id": group_id, "n": len(positions),
            "mae": math.fsum(errors) / len(errors),
            "spearman": spearman([gold[i] for i in positions], [values[i] for i in positions])})
    return {"n": len(rows), "mae": math.fsum(abs(y-p) for y,p in zip(gold,values))/len(rows) if rows else None,
            "spearman": spearman(gold,values), "n_groups": len(per_group),
            "group_macro_mae": math.fsum(g["mae"] for g in per_group)/len(per_group) if per_group else None,
            "groups": per_group}


def generation_metrics(rows, predictions):
    expected = [r["sample_id"] for r in rows]
    actual = [r.get("sample_id") for r in predictions]
    if actual != expected or len(set(actual)) != len(actual):
        raise ValueError("Predictions must exactly cover frozen test IDs in original order")
    valid_rows, values, failures = [], [], Counter()
    for row, prediction in zip(rows,predictions):
        if prediction.get("split") != "test" or not finite(prediction.get("gold_pkd")) or prediction["gold_pkd"] != row["pkd"]:
            raise ValueError("Prediction split/gold differs from frozen test")
        parsed = parse_answer(prediction.get("raw_text"))
        saved = prediction.get("parse")
        if (not isinstance(saved,dict) or saved.get("valid") is not parsed["valid"]
                or saved.get("reason") != parsed["reason"] or set(saved) != {"valid","value","reason"}
                or (parsed["valid"] and (not finite(saved.get("value")) or saved["value"] != parsed["value"]))
                or (not parsed["valid"] and saved.get("value") is not None)):
            raise ValueError("Stored parse disagrees with independent original strict parser")
        if parsed["valid"]:
            valid_rows.append(row); values.append(parsed["value"])
        else:
            failures[parsed["reason"]] += 1
    metric = regression(valid_rows,values)
    return {"n_total":len(rows),"n_valid":len(values),"n_failed":len(rows)-len(values),
            "parse_valid_rate":len(values)/len(rows) if rows else None,"failure_counts":dict(failures),
            "full_sample_mae":metric["mae"] if len(values)==len(rows) else None,
            "valid_only_mae":metric["mae"],"valid_only_denominator":len(values),"regression_valid_only":metric}


def seed_statistics(per_seed):
    if set(per_seed) != {str(s) for s in SEEDS} or any(v is not None and not finite(v) for v in per_seed.values()):
        raise ValueError("All three explicit seed keys are required; values must be finite or null")
    missing = [seed for seed in SEEDS if per_seed[str(seed)] is None]
    values = [per_seed[str(seed)] for seed in SEEDS]
    complete = not missing
    return {"k":3-len(missing),"expected_k":3,"coverage":f"{3-len(missing)}/3","per_seed":per_seed,
            "mean":statistics.mean(values) if complete else None,
            "sample_SD":statistics.stdev(values) if complete else None,"missing_invalid_seeds":missing}


def official_eligible(summary_status, acceptance, raw_consistent):
    """Independent historical acceptance is mandatory even when result says complete."""
    return summary_status == "complete" and acceptance is True and raw_consistent


def case_key(case):
    return tuple(case.get(k) for k in ("scenario","seed","placement","message_input","sender_mode","gate_init"))


def expected_matrix():
    cases=[]
    for seed in SEEDS:
        for scenario in SCENARIOS:
            for placement in ("early","mid"):
                for content in ("real","neutral"):
                    cases.append(dict(scenario=scenario,seed=seed,placement=placement,message_input=content,sender_mode="balanced",gate_init="zero",phase="trial"))
    for extension in ("near_unit","global16"):
        for seed in SEEDS:
            for scenario in SCENARIOS:
                cases.append(dict(scenario=scenario,seed=seed,placement="mid",message_input="real",
                    sender_mode="global16" if extension=="global16" else "balanced",
                    gate_init="near_unit" if extension=="near_unit" else "zero",phase="trial"))
    return cases


def identifier(case):
    return "trial_"+"_".join(str(case[k]) for k in ("scenario","seed","placement","message_input","sender_mode","gate_init"))


class Audit:
    def __init__(self,root):
        self.root=Path(root).resolve(); self.errors=[]; self.input_sha256={}; self.checks=0

    def check(self,condition,message):
        self.checks+=1
        if not condition:self.errors.append(message)
        return bool(condition)

    def read(self,relative,jsonl=False):
        p=(self.root/relative).resolve()
        if not p.is_relative_to(self.root):raise ValueError("Input escapes repository root")
        data=p.read_bytes();self.input_sha256[p.relative_to(self.root).as_posix()]=hashlib.sha256(data).hexdigest()
        text=data.decode("utf-8")
        return [strict_loads(line)for line in text.splitlines() if line.strip()] if jsonl else strict_loads(text)

    def same(self,actual,expected,label):
        if expected is None or actual is None:
            return self.check(actual is None and expected is None,label)
        if type(actual)is bool or type(expected)is bool:
            return self.check(type(actual)is bool and type(expected)is bool and actual is expected,label)
        if finite(actual) and finite(expected):
            return self.check(math.isclose(actual,expected,rel_tol=ATOL,abs_tol=ATOL),label)
        return self.check(actual==expected,label)


def audit_training(audit,history,selection,result,label):
    audit.check(len(history)==3,label+": exactly three epochs")
    if len(history)!=3:return
    audit.check([h.get("epoch")for h in history]==[1,2,3],label+": epoch sequence")
    audit.check([h.get("optimizer_steps")for h in history]==[516,1032,1548],label+": optimizer updates")
    if not audit.check(all(finite(h.get("validation_answer_token_nll"))for h in history),label+": finite validation NLL"):
        return
    selected=min(history,key=lambda h:h["validation_answer_token_nll"])
    for obj,key in [(result,"best_validation_answer_token_nll"),(selection,"validation_answer_token_nll")]:
        audit.same(obj.get(key),selected["validation_answer_token_nll"],label+": original validation selection value")
        audit.check(obj.get("best_epoch")==selected["epoch"],label+": earliest best validation epoch")
    audit.check(result.get("actual_epochs")==3 and result.get("optimizer_steps")==1548,label+": result full training budget")
    audit.check(selection.get("all_training_epochs_completed")==3 and selection.get("optimizer_steps")==1548,label+": selection budget")
    audit.check(selection.get("test_labels_used_for_selection") is False,label+": no test selection")


def compare_regression(audit,computed,stored,label):
    if not isinstance(stored,dict):audit.check(False,label+": stored regression missing");return
    for key in ("n","mae","spearman","n_groups","group_macro_mae"):
        audit.same(computed[key],stored.get(key),label+": "+key)
    saved=stored.get("groups")
    if not audit.check(isinstance(saved,list)and len(saved)==len(computed["groups"]),label+": group coverage"):return
    for actual,expected in zip(computed["groups"],saved):
        audit.check(actual["group_id"]==expected.get("group_id"),label+": group order/ID")
        for key in ("n","mae","spearman"):audit.same(actual[key],expected.get(key),label+": group "+key)


def audit_c1(audit,rows,records,result,label):
    audit.check([r.get("sample_id")for r in records]==[r["sample_id"]for r in rows],label+": C1 exact test coverage")
    if not records:return None
    correct,off=[],[];contributions={name:[] for name in ("format","numeric","mixed")}
    for item in records:
        c,o=item.get("correct_original_gold_mean_token_nll"),item.get("off_original_gold_mean_token_nll")
        n=item.get("answer_tokens")
        if not audit.check(finite(c)and finite(o)and type(n)is int and n>0,label+": finite row C1/count"):
            return None
        correct.append(c);off.append(o)
        breakdown=item.get("token_breakdown",{})
        for mode in ("correct","off"):
            b=breakdown.get(mode,{})
            if not audit.check(set(b)=={"format","numeric","mixed","full"},label+": C1 category fields"):return None
            audit.check(b["full"].get("count")==n and sum(b[k].get("count",-1)for k in contributions)==n,label+": C1 category denominator")
            for k in (*contributions,"full"):
                loss=b[k].get("loss_sum");count=b[k].get("count")
                if not audit.check(finite(loss)and type(count)is int and count>=0,label+": finite C1 category sum/count"):return None
                audit.same(b[k].get("contribution_nats_per_full_token"),loss/n,label+": C1 shared full denominator")
                audit.same(b[k].get("nll_nats_per_token"),loss/count if count else None,label+": C1 category mean")
        for k in contributions:
            contributions[k].append((breakdown["off"][k]["loss_sum"]-breakdown["correct"][k]["loss_sum"])/n)
    n=len(records);means={"correct_original_gold_mean_token_nll":math.fsum(correct)/n,
                        "off_original_gold_mean_token_nll":math.fsum(off)/n}
    gain=math.fsum(o-c for o,c in zip(off,correct))/n
    original=result.get("C1",{})
    audit.check(original.get("n_test")==n,label+": C1 n_test")
    audit.same(original.get("G_all_test"),gain,label+": original row-mean C1 G")
    for key,value in means.items():audit.same(original.get("means_all_test",{}).get(key),value,label+": C1 mean")
    decomposition={key:math.fsum(values)/n for key,values in contributions.items()}
    for key,value in decomposition.items():audit.same(original.get("G_decomposition",{}).get(key),value,label+": original C1 category gain")
    residual=gain-sum(decomposition.values())
    audit.same(original.get("G_decomposition",{}).get("reduction_rounding_residual"),residual,label+": preserved C1 reduction residual")
    return {"n_test":n,"G_all_test":gain,"means_all_test":means,"G_decomposition":decomposition,
            "reduction_rounding_residual":residual,"meaning":"stored teacher-forced branch effect, not molecular information"}


def load_datasets(audit):
    datasets={}
    for scenario in SCENARIOS:
        base=Path("data/a1")/scenario
        rows=audit.read(base/"samples.jsonl",True);gold=audit.read(base/"test_gold.jsonl",True)
        lock=audit.read(base/"benchmark_lock.json");audit.read(base/"manifest.json")
        for filename,expected in DATA_SHA256[scenario].items():
            audit.check(audit.input_sha256[(base/filename).as_posix()]==expected,scenario+": frozen "+filename+" SHA")
        audit.check(Counter(r.get("split")for r in rows)==COUNTS[scenario],scenario+": frozen split counts")
        audit.check(len({r.get("sample_id")for r in rows})==len(rows),scenario+": unique sample IDs")
        test=[r for r in rows if r.get("split")=="test"]
        audit.check([r.get("sample_id")for r in gold]==[r["sample_id"]for r in test],scenario+": test_gold order")
        audit.check(all(finite(r.get("pkd"))and r["pkd"]==g.get("pkd")for r,g in zip(test,gold)),scenario+": original unrounded gold")
        audit.check(lock.get("split_counts")==COUNTS[scenario],scenario+": lock split counts")
        datasets[scenario]=test
    return datasets


def build_statistics(official):
    by_key={case_key(row["case"]):row["raw_metrics"]["full_sample_mae"] for row in official}
    def value(s,seed,p,content="real",sender="balanced",gate="zero"):
        return by_key.get((s,seed,p,content,sender,gate))
    configurations=[];contrasts=[]
    conditions=[("early","real","balanced","zero"),("early","neutral","balanced","zero"),
                ("mid","real","balanced","zero"),("mid","neutral","balanced","zero"),
                ("mid","real","balanced","near_unit"),("mid","real","global16","zero")]
    for scenario in SCENARIOS:
        for p,c,sender,gate in conditions:
            configurations.append({"scenario":scenario,"configuration":[p,c,sender,gate],
                **seed_statistics({str(seed):value(scenario,seed,p,c,sender,gate)for seed in SEEDS})})
        for contrast in CONTRASTS:
            per_seed={}
            for seed in SEEDS:
                er,en,mr,mn=(value(scenario,seed,p,c)for p,c in [("early","real"),("early","neutral"),("mid","real"),("mid","neutral")])
                pairs={"depth_real":(er,mr),"depth_neutral":(en,mn),"content_early":(en,er),"content_mid":(mn,mr),
                       "gate_zero_initialization":(value(scenario,seed,"mid",gate="near_unit"),mr),
                       "balanced_pool_allocation":(value(scenario,seed,"mid",sender="global16"),mr)}
                if contrast=="depth_content_difference_in_differences":
                    number=(er-mr)-(en-mn) if all(x is not None for x in (er,en,mr,mn))else None
                else:
                    a,b=pairs[contrast];number=a-b if a is not None and b is not None else None
                per_seed[str(seed)]=number
            contrasts.append({"scenario":scenario,"name":contrast,**seed_statistics(per_seed)})
    return configurations,contrasts


def compare_statistics(audit,recomputed,stored,configuration=False):
    key=lambda x:(x.get("scenario"),tuple(x.get("configuration",[]))if configuration else x.get("name"))
    lookup={key(row):row for row in stored}
    audit.check(len(lookup)==len(stored)==len(recomputed),"unique complete configuration/contrast table")
    for row in recomputed:
        saved=lookup.get(key(row),{})
        for k in ("k","expected_k","coverage","missing_invalid_seeds"):
            audit.check(row[k]==saved.get(k),str(key(row))+": "+k)
        for seed in SEEDS:audit.same(row["per_seed"][str(seed)],saved.get("per_seed",{}).get(str(seed)),str(key(row))+": seed"+str(seed))
        for k in ("mean","sample_SD"):audit.same(row[k],saved.get(k),str(key(row))+": "+k)


def audit_repository(root):
    audit=Audit(root)
    try:
        summary=audit.read("results/matched_budget/final_summary.json")
        datasets=load_datasets(audit)
        expected=expected_matrix();expected_ids={identifier(c):c for c in expected}
        source_cases=summary.get("cases",[]);source={row.get("case_id"):row for row in source_cases}
        audit.check(summary.get("schema_version")==1 and summary.get("n_expected_training_trials")==36,"fixed summary schema/campaign size")
        audit.check(summary.get("protocol_sha256")=="9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208","frozen scientific protocol identity")
        audit.check(len(source_cases)==len(source)==36 and set(source)==set(expected_ids),"exact unique frozen36 case IDs")
        audit.check(summary.get("evaluation_version")=="original_V1_unchanged_batch1_all36","originalV1 scientific channel")
        case_root=audit.root/"results/matched_budget/cases"
        actual_dirs={p.name for p in case_root.iterdir()if p.is_dir()}
        audit.check(actual_dirs<=set(expected_ids),"no unknown actual attempt directory")
        attempts=[];missing=[]
        for case_id,configuration in expected_ids.items():
            record=source.get(case_id,{})
            audit.check(record.get("case")==configuration,case_id+": frozen case configuration")
            if case_id not in actual_dirs:
                missing.append(case_id)
                audit.check(record.get("status")=="missing"and record.get("metrics")is None,case_id+": source missing state")
                continue
            errors_before=len(audit.errors);base=Path("results/matched_budget/cases")/case_id
            case=audit.read(base/"case.json");result=audit.read(base/"result.json")
            acceptance=audit.read(base/"acceptance.json")
            history=audit.read(base/"history.json");selection=audit.read(base/"selection_before_test.json")
            predictions=audit.read(base/"test_predictions_correct.jsonl",True)
            c1_records=audit.read(base/"test_C1_answer_nll.jsonl",True)
            audit.check(case==configuration,case_id+": actual case identity")
            accepted=acceptance.get("valid_for_acceptance")
            audit.check(type(accepted)is bool,case_id+": independent acceptance boolean required")
            if "case_id"in acceptance:audit.check(acceptance["case_id"]==case_id,case_id+": acceptance identity")
            if "official_summary_status"in acceptance:
                audit.check(acceptance["official_summary_status"]==record.get("status"),case_id+": acceptance summary status")
            if "original_validation_sha256"in acceptance:
                pin=acceptance["original_validation_sha256"]
                audit.check(isinstance(pin,str)and len(pin)==64 and set(pin.lower())<=set("0123456789abcdef"),case_id+": original validation SHA format")
            audit.check(result.get("status")=="complete",case_id+": stored attempt result status")
            audit_training(audit,history,selection,result,case_id)
            metrics=generation_metrics(datasets[configuration["scenario"]],predictions)
            saved=result.get("test_generation",{}).get("correct",{})
            for key in ("n_total","n_valid","n_failed","parse_valid_rate","failure_counts","full_sample_mae"):
                audit.same(metrics[key],saved.get(key),case_id+": result "+key)
            compare_regression(audit,metrics["regression_valid_only"],saved.get("regression_valid_only"),case_id+": result regression")
            c1=audit_c1(audit,datasets[configuration["scenario"]],c1_records,result,case_id)
            if record.get("status")=="complete":
                audit.check(accepted is True,case_id+": official source requires independent acceptance")
                public=record.get("metrics",{})
                for key in ("n_total","n_valid","n_failed","full_sample_mae","valid_only_mae","valid_only_denominator"):
                    audit.same(metrics[key],public.get(key),case_id+": final summary "+key)
                compare_regression(audit,metrics["regression_valid_only"],public.get("target_macro"),case_id+": summary regression")
            else:
                audit.check(accepted is False and record.get("metrics")is None,case_id+": quarantine has no official metric")
                audit.check(record.get("status")=="invalid_or_incomplete",case_id+": quarantined historical state")
            consistent=len(audit.errors)==errors_before
            attempts.append({"case_id":case_id,"case":configuration,"summary_status":record.get("status"),
                "valid_for_acceptance":accepted,"acceptance":acceptance,"raw_consistent":consistent,
                "included_in_official_tables":official_eligible(record.get("status"),accepted,consistent),
                "raw_metrics":metrics,"stored_C1_recomputed":c1})
        official=[a for a in attempts if a["included_in_official_tables"]]
        quarantine=[a for a in attempts if not a["included_in_official_tables"]]
        configurations,contrasts=build_statistics(official)
        compare_statistics(audit,configurations,summary.get("seed_summaries",[]),True)
        compare_statistics(audit,contrasts,summary.get("paired_contrasts",[]))
        n_prediction=sum(a["raw_metrics"]["n_total"]for a in official)
        n_valid=sum(a["raw_metrics"]["n_valid"]for a in official)
        audit.check(len(attempts)==31 and len(official)==30 and len(quarantine)==1 and len(missing)==5,"release disposition30accepted/1quarantined/5missing")
        audit.check(summary.get("n_complete")==len(official)==30,"official completed case count")
        audit.check(n_prediction==n_valid==8910,"all8910 official saved predictions reconstructed")
        audit.check(sum(a["raw_metrics"]["n_total"]for a in quarantine)==152,"quarantine152 rows remain separate")
        report={"validation_passed":not audit.errors,"checks":audit.checks,"errors":audit.errors,
            "source_summary_sha256":audit.input_sha256["results/matched_budget/final_summary.json"],
            "input_sha256":audit.input_sha256,"attempts":attempts,"official_case_ids":[a["case_id"]for a in official],
            "quarantined_case_ids":[a["case_id"]for a in quarantine],"missing_case_ids":missing,
            "counts":{"expected":36,"raw_attempts":len(attempts),"official":len(official),"quarantined":len(quarantine),"missing":len(missing),
                "official_predictions":n_prediction,"official_parse_valid":n_valid,
                "quarantined_predictions":sum(a["raw_metrics"]["n_total"]for a in quarantine)},
            "seed_summaries":configurations,"paired_contrasts":contrasts,
            "interpretation":{"sample_SD":"paired seed variability, not CI; requires all3 finite seeds",
                "C1":"saved teacher-forced branch effect, not molecular information",
                "quarantine":"numerical reconstruction does not override historical acceptancefalse",
                "token_ids_redecoded":False,"models_loaded":False,"network_requests":0,"training_or_fit_performed":False,
                "historical_window_revalidated_against_today":False}}
        return report
    except (OSError,ValueError,TypeError,KeyError,OverflowError,RecursionError)as error:
        return {"validation_passed":False,"checks":audit.checks,"errors":audit.errors+[type(error).__name__+": "+str(error)],
                "input_sha256":audit.input_sha256}


def export_report(report,output,root):
    output=Path(output).resolve();root=Path(root).resolve()
    if output==root or root.is_relative_to(output)or any(output.is_relative_to(root/name)for name in("data","results","publication")):
        raise ValueError("Choose a separate output directory outside frozen data/results/publication")
    if output.exists()and any(output.iterdir()):raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True,exist_ok=True)
    with(output/"analysis.json").open("x",encoding="utf-8")as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    tables={"cases":[],"seed_summaries":[],"paired_contrasts":[]}
    for a in report.get("attempts",[]):
        m=a["raw_metrics"];tables["cases"].append({"case_id":a["case_id"],**a["case"],"summary_status":a["summary_status"],
            "valid_for_acceptance":a["valid_for_acceptance"],"raw_consistent":a["raw_consistent"],"official":a["included_in_official_tables"],
            "n_total":m["n_total"],"n_valid":m["n_valid"],"n_failed":m["n_failed"],"raw_full_sample_mae":m["full_sample_mae"],
            "official_full_sample_mae":m["full_sample_mae"]if a["included_in_official_tables"]else None,
            "valid_only_denominator":m["valid_only_denominator"],
            "raw_valid_only_spearman":m["regression_valid_only"]["spearman"]})
    for table in ("seed_summaries","paired_contrasts"):
        for row in report.get(table,[]):
            flat={k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in row.items()if k!="per_seed"}
            flat.update({"seed_"+str(seed):row["per_seed"][str(seed)]for seed in SEEDS});tables[table].append(flat)
    for name,rows in tables.items():
        if not rows:continue
        with(output/(name+".csv")).open("x",encoding="utf-8",newline="")as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for name in("validate","analyze"):
        p=commands.add_parser(name);p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1])
        if name=="analyze":p.add_argument("--output", "--output-dir", dest="output_dir", type=Path, required=True)
    args=parser.parse_args(argv);report=audit_repository(args.repo_root)
    if args.command=="analyze":export_report(report,args.output_dir,args.repo_root)
    print(json.dumps({"validation_passed":report["validation_passed"],"checks":report["checks"],
                      "counts":report.get("counts"),"errors":report["errors"]},ensure_ascii=False,allow_nan=False))
    return 0 if report["validation_passed"]else 1


if __name__=="__main__":
    raise SystemExit(main())
