"""Capacity-limited supportive outreach; never an eligibility decision."""
import io
import numpy as np
import pandas as pd


def plan_outreach(scored_frame: pd.DataFrame, capacity: int) -> pd.DataFrame:
    if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity < 0:
        raise ValueError('Capacity must be a non-negative whole number.')
    if 'miss_probability' not in scored_frame or 'appointment_id' not in scored_frame:
        raise ValueError('Scores and appointment references are required.')
    p = pd.to_numeric(scored_frame.miss_probability, errors='coerce')
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError('All probabilities must be finite and between zero and one.')
    result = scored_frame.assign(miss_probability=p).sort_values(
        ['miss_probability', 'appointment_id'], ascending=[False, True], kind='stable'
    ).head(int(capacity)).copy().reset_index(drop=True)
    result.insert(0, 'priority', np.arange(1, len(result) + 1))
    result['review_status'] = 'Needs staff review'
    return result


def export_csv(frame: pd.DataFrame) -> bytes:
    """Escape spreadsheet formula prefixes without changing numeric scores."""
    safe = frame.copy()
    for col in safe.select_dtypes(include=['object', 'string']).columns:
        safe[col] = safe[col].map(lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else x)
    return safe.to_csv(index=False).encode('utf-8-sig')
