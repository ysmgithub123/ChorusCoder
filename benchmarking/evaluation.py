from benchmarking.metrics import aggregate_scores
from pathlib import Path
from benchmarking.common import AtomicJsonlWriter, read_jsonl, task_id_from_reference, write_json

def _parser(language):
    if language != 'python':
        raise ValueError('The paper benchmarks use Python')
    from choruscoder import make_parser
    return make_parser(language)

def _id_scores(prediction, target, language):
    from utils.eval_utils import extract_identifiers
    pred_id_list = extract_identifiers(prediction, language)
    target_id_list = extract_identifiers(target, language)
    pred_ids = set(pred_id_list)
    target_ids = set(target_id_list)
    true_positive = len(pred_ids & target_ids)
    false_positive = len(pred_ids - target_ids)
    false_negative = len(target_ids - pred_ids)
    denominator = 2 * true_positive + false_positive + false_negative
    return (int(pred_id_list == target_id_list), 2 * true_positive / denominator if denominator else 0.0)

def _repository_id(task_id, reference):
    metadata = reference.get('metadata') or {}
    return str(metadata.get('repository') or metadata.get('repo') or str(task_id).split('/', 1)[0])

def evaluate_predictions(prediction_file, reference_file, language, output_dir, require_complete=True, benchmark='cceval'):
    predictions = read_jsonl(prediction_file)
    references = read_jsonl(reference_file)
    prediction_map = {str(row['task_id']): row for row in predictions}
    reference_map = {task_id_from_reference(row): row for row in references}
    if len(prediction_map) != len(predictions):
        raise ValueError('prediction.jsonl contains duplicate task_id values')
    if len(reference_map) != len(references) or '' in reference_map:
        raise ValueError('Evaluation reference contains missing or duplicate task_id values')
    unexpected = sorted(set(prediction_map) - set(reference_map))
    missing = sorted(set(reference_map) - set(prediction_map))
    if unexpected or (require_complete and missing):
        raise ValueError(f'Prediction/reference task IDs differ: missing={len(missing)}, unexpected={len(unexpected)}. Examples: missing={missing[:3]}, unexpected={unexpected[:3]}')
    reference_map = {task_id: reference_map[task_id] for task_id in prediction_map}
    del predictions, references
    if benchmark == 'recceval':
        if language != 'python':
            raise ValueError('ReccEval evaluation is Python-only')
        from benchmarking.recceval_metrics import score_completion, EVALUATION_VERSION
    else:
        from utils.eval_utils import cal_edit_sim, postprocess_code_lines, remove_comments
    parser = None if benchmark == 'recceval' else _parser(language)
    score_rows = []
    with AtomicJsonlWriter(Path(output_dir) / 'evaluation_details.jsonl') as detail_writer:
        for task_id, reference in reference_map.items():
            raw_prediction = str(prediction_map[task_id].get('pred', ''))
            if benchmark == 'recceval':
                score = dict(task_id=task_id, repository=_repository_id(task_id, reference), raw_pred=raw_prediction, evaluation_version=EVALUATION_VERSION, **score_completion(raw_prediction, str(reference['groundtruth'])))
                score_rows.append(score)
                detail_writer.write(score)
                continue
            prompt = str(reference.get('prompt', ''))
            target = remove_comments(str(reference['groundtruth']))
            prediction = remove_comments(postprocess_code_lines(prompt, raw_prediction, parser, language))
            pred_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
            target_lines = [line.strip() for line in target.splitlines() if line.strip()]
            code_em = int(pred_lines == target_lines)
            code_es = cal_edit_sim([target], [prediction])
            identifier_em, identifier_f1 = _id_scores(prediction, target, language)
            score = {'task_id': task_id, 'repository': _repository_id(task_id, reference), 'pred': prediction, 'target': target, 'code_em': code_em, 'code_es': code_es, 'identifier_em': identifier_em, 'identifier_f1': identifier_f1}
            score_rows.append(score)
            detail_writer.write(score)
    metrics = aggregate_scores(score_rows)
    if benchmark == 'recceval':
        metrics.update(evaluation_version=EVALUATION_VERSION, aggregation='sample_micro')
    output_dir = Path(output_dir)
    write_json(output_dir / 'metrics.json', metrics)
    return metrics

def aggregate_timings(timing_file):
    rows = read_jsonl(timing_file)
    if not rows:
        return {'retrieval_seconds_per_sample': 0.0, 'generation_seconds_per_sample': 0.0, 'end_to_end_seconds_per_sample': 0.0}
    return {'retrieval_seconds_per_sample': round(sum((float(x['retrieval_seconds']) for x in rows)) / len(rows), 6), 'generation_seconds_per_sample': round(sum((float(x['generation_seconds']) for x in rows)) / len(rows), 6), 'end_to_end_seconds_per_sample': round(sum((float(x['end_to_end_seconds']) for x in rows)) / len(rows), 6)}
