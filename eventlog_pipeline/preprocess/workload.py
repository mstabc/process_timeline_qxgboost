import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# TODO: it needs an optimization (Joblib maybe) It has two nested loops over all the days (from min date to max date) and all the tasks which can be slow for large datasets
def compute_workload(df: pd.DataFrame):
    queue_records, progress_records = [], []
    
    # Log overall date range and tasks
    min_date = df['start_date'].min()
    max_date = df['complete_date'].max()
    all_dates = pd.date_range(min_date, max_date, freq='D')
    tasks = sorted(df['task'].unique())
    logging.info(f"Computing workload from {min_date.date()} to {max_date.date()}")
    logging.info(f"Identified {len(tasks)} unique tasks")

    for date in all_dates:
        
        queue = {'update_date': date}
        progress = {'update_date': date}
        in_queue_cols = []
        in_progress_cols = []
        logging.debug(f"Processing date: {date.date()}")

        for task in tasks:
            task_df = df[df['task'] == task]
            
            #How many tasks are in queue and in progress on this date per task 
            queue_count = len(task_df[(task_df['create_date'] < date) & (task_df['start_date'] > date)])
            progress_count = len(task_df[(task_df['start_date'] <= date) & (task_df['complete_date'] > date)])
            
            queue[f'{task}_in_queue'] = queue_count
            in_queue_cols.append(f'{task}_in_queue')
            
            progress[f'{task}_in_progress'] = progress_count
            in_progress_cols.append(f'{task}_in_progress')
            
            logging.debug(f"Task '{task}': queue={queue_count}, progress={progress_count}")

        queue_records.append(queue)
        progress_records.append(progress)

    logging.info(f"Finished computing workload for {len(all_dates)} days")
    




    return pd.DataFrame(queue_records), in_queue_cols, pd.DataFrame(progress_records), in_progress_cols


