import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def compute_workload(df: pd.DataFrame):
    """
    Compute per-day workload per task in a vectorized way.

    For each task:
      - *_in_queue:   create_date < date < start_date
      - *_in_progress: start_date <= date < complete_date
    """

    df = df.copy()
    df["create_date"] = pd.to_datetime(df["create_date"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["complete_date"] = pd.to_datetime(df["complete_date"])

    min_date = df["start_date"].min().normalize()
    max_date = df["complete_date"].max().normalize()
    all_dates = pd.date_range(min_date, max_date, freq="D")
    date_index = all_dates.to_numpy()

    tasks = sorted(df["task"].unique())
    logging.info("Computing workload from %s to %s", min_date.date(), max_date.date())
    logging.info("Identified %d unique tasks", len(tasks))

    # We will build the result column-by-column instead of row-by-row
    queue_data = {"update_date": all_dates}
    progress_data = {"update_date": all_dates}
    in_queue_cols = []
    in_progress_cols = []

    for task in tasks:
        task_df = df[df["task"] == task]
        if task_df.empty:
            # No instances of this task
            queue_col_name = f"{task}_in_queue"
            progress_col_name = f"{task}_in_progress"
            queue_data[queue_col_name] = np.zeros(len(all_dates), dtype=int)
            progress_data[progress_col_name] = np.zeros(len(all_dates), dtype=int)
            in_queue_cols.append(queue_col_name)
            in_progress_cols.append(progress_col_name)
            continue

        create = task_df["create_date"].to_numpy()
        start = task_df["start_date"].to_numpy()
        complete = task_df["complete_date"].to_numpy()

        # Queue: create_date < d < start_date
        # first index with date > create_date
        q_start_idx = date_index.searchsorted(create, side="right")
        # first index with date >= start_date (we want d < start_date)
        q_end_idx = date_index.searchsorted(start, side="left")

        delta_q = np.zeros(len(all_dates) + 1, dtype=int)
        mask_q = q_start_idx < q_end_idx
        np.add.at(delta_q, q_start_idx[mask_q], 1)
        np.add.at(delta_q, q_end_idx[mask_q], -1)
        queue_counts = delta_q[:-1].cumsum()

        # In progress: start_date <= d < complete_date
        p_start_idx = date_index.searchsorted(start, side="left")
        p_end_idx = date_index.searchsorted(complete, side="left")

        delta_p = np.zeros(len(all_dates) + 1, dtype=int)
        mask_p = p_start_idx < p_end_idx
        np.add.at(delta_p, p_start_idx[mask_p], 1)
        np.add.at(delta_p, p_end_idx[mask_p], -1)
        progress_counts = delta_p[:-1].cumsum()

        queue_col_name = f"{task}_in_queue"
        progress_col_name = f"{task}_in_progress"

        queue_data[queue_col_name] = queue_counts
        progress_data[progress_col_name] = progress_counts

        in_queue_cols.append(queue_col_name)
        in_progress_cols.append(progress_col_name)

        logging.debug(
            "Finished task '%s': max queue=%d, max in_progress=%d",
            task,
            queue_counts.max(),
            progress_counts.max(),
        )

    queue_df = pd.DataFrame(queue_data)
    progress_df = pd.DataFrame(progress_data)

    logging.info(
        "Finished computing workload for %d days and %d tasks",
        len(all_dates),
        len(tasks),
    )

    return queue_df, in_queue_cols, progress_df, in_progress_cols
