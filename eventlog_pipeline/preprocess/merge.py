import pandas as pd
from sklearn.preprocessing import LabelEncoder
from eventlog_pipeline.utils.helpers import drop_cols

#This function floors datetime series to day level as the timestamps are in hour and minute level, For workload merges we need to do it on day level
def _floor_day(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.floor('D')

def merger(event_log_df, workload_in_queue_df, workload_in_progress_df, case_summary_df):
    
    event_log_df['create_date_only'] = _floor_day(event_log_df['create_date'])
    workload_in_queue_df['update_date_only'] = _floor_day(workload_in_queue_df['update_date'])
    workload_in_progress_df['update_date_only'] = _floor_day(workload_in_progress_df['update_date'])

    # Merge in queue ones
    merged = pd.merge(
        event_log_df,
        workload_in_queue_df,
        left_on='create_date_only',
        right_on='update_date_only',
        how='left',
        suffixes=('', '_in_queue')
    )
    
    
    # Merge in progress ones
    merged = pd.merge(
        merged,
        workload_in_progress_df,
        left_on='create_date_only',
        right_on='update_date_only',
        how='left',
        suffixes=('', '_in_progress')
    )


    # Merge case summary
    merged = pd.merge(
        merged,
        case_summary_df,
        left_on='case_id',
        right_on='case_id',
        how='left',
        suffixes=('', '_exists')
    )
    
    
    drop_cols(merged, ['create_date_only', 'update_date_only', 'update_date_in_progress'])

    label_encoder = LabelEncoder()
    merged['task_encoded'] = label_encoder.fit_transform(merged['task'].astype(str))

    return merged, label_encoder
