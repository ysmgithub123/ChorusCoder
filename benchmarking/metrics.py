"""Lightweight benchmark-specific aggregation shared by evaluation and analysis."""

def aggregate_scores(score_rows):
    """Sample-level means used by CrossCodeEval and ReccEval."""
    if not score_rows:
        raise ValueError('No prediction/reference pairs were available for evaluation')

    def average(field, scale=1.0):
        return sum((float(row[field]) / scale for row in score_rows)) / len(score_rows) * scale
    return dict(total_samples=len(score_rows), code_em=round(average('code_em') * 100, 4), code_es=round(average('code_es', 100.0), 4), identifier_em=round(average('identifier_em') * 100, 4), identifier_f1=round(average('identifier_f1') * 100, 4))
