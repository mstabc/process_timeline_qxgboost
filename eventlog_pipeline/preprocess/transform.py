import pandas as pd
import logging
from eventlog_pipeline.utils.helpers import _task_before, _case_before


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def compute_durations_and_sequences(df):
    
    logging.info(f"building durations and sequences for {len(df['case_id'].unique())} unique cases and {len(df)} events")
    # Create & Start & Complete dates
    df['complete_date'] = df['timestamp']
    df['create_date'] = df.groupby('case_id')['timestamp'].transform('min')
    df['start_date'] = df.groupby('case_id')['timestamp'].shift(1).fillna(df['timestamp']) #Basically, the start date of the first task can not be retrieved as we don't know when the case was created, so we use the complete date of the first task as its start date.   
    df['case_complete_date'] = df.groupby('case_id')['complete_date'].transform('max')
    
    # Task and Case Duration
    df['duration'] = df['complete_date'] - df['start_date']
    df['duration_sec'] = df['duration'].dt.total_seconds().replace(0, 1)

    df['total_duration'] = df.groupby('case_id')['complete_date'].transform('max') - df['create_date']
    df['total_duration_sec'] = df['total_duration'].dt.total_seconds().replace(0, 1)

    #Create date parts
    df['created_day_of_month'] = df['create_date'].dt.day
    df['created_week_of_year'] = df['create_date'].dt.isocalendar().week
    df['created_month_of_year'] = df['create_date'].dt.month
    df['created_year'] = df['create_date'].dt.year
    
    # Relative Start & Complete (as fraction of total case duration)
    df['relative_start'] = (df['start_date'] - df['create_date']).dt.total_seconds() / df['total_duration_sec']
    df['relative_complete'] = (df['complete_date'] - df['create_date']).dt.total_seconds() / df['total_duration_sec']
    logging.info(f"Something to check they must be <=1::: Relative Start: {max(df['relative_start'])}, Relative Complete: {max(df['relative_complete'])}")
    logging.info("Completed building durations and sequences.")
    
    return df

#For sure needs an optimization!!
def prior_completions(df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"building prior completions for {len(df)} events")
    """
    For each row:
      - Count how many same tasks were completed BEFORE this task's create_date
        in the same date/week/month/year bucket.
      - Count how many unique cases were completed BEFORE this case's create_date
        in the same date/week/month/year bucket.
    """

    df = df.copy()
    df = df.sort_values(["create_date", "case_id"]).reset_index(drop=True)

    #Time buckets
    same_day = lambda x: x.date()
    same_week = lambda x: (x.isocalendar().year, x.isocalendar().week)
    same_month = lambda x: (x.year, x.month)
    same_year = lambda x: x.year

    df = _task_before(df, same_day, "task_completed_before_same_day")
    df = _task_before(df, same_week, "task_completed_before_same_week")
    df = _task_before(df, same_month, "task_completed_before_same_month")
    df = _task_before(df, same_year, "task_completed_before_same_year")

    df = _case_before(df, same_day, "cases_completed_before_same_day")
    df = _case_before(df, same_week, "cases_completed_before_same_week")
    df = _case_before(df, same_month, "cases_completed_before_same_month")
    df = _case_before(df, same_year, "cases_completed_before_same_year")
    
    in_troughput_columns = [
        "task_completed_before_same_day",
        "task_completed_before_same_week",
        "task_completed_before_same_month",
        "task_completed_before_same_year",
        "cases_completed_before_same_day",
        "cases_completed_before_same_week",
        "cases_completed_before_same_month",
        "cases_completed_before_same_year",
    ]
    
    logging.info("Completed building prior_completions over time.")
    return df, in_troughput_columns

# To know what tasks were involved in each case
def create_case_summary(df):
    summaries = []
    unique_tasks = sorted(df['task'].unique())
    
    for case_id, group in df.groupby('case_id'):
        create = group['create_date'].min()
        end = group['complete_date'].max()
        total_duration = group['total_duration_sec']

        summary = {
            'case_id': case_id,
            'total_duration': total_duration,
            # These date parts are useful for aggregations if total duration is needed
            'created_day_of_month': create.day,
            'created_week_of_year': create.isocalendar()[1],
            'created_month_of_year': create.month,
            'created_year': create.year,
            
            'completed_day_of_month': end.day,
            'completed_week_of_year': end.isocalendar()[1],
            'completed_month_of_year': end.month,
            'completed_year': end.year
        }

        # Add one column per task name, with 1 if present in the case
        exists_columns = []
        present_tasks = set(group['task'].values)
        exists_columns = [f"{task}_exists" for task in present_tasks]
        for task in unique_tasks:
            summary[task] = int(task in present_tasks)

        summaries.append(summary)
    logging.info(f"Completed building case summary.")
    return pd.DataFrame(summaries), exists_columns
