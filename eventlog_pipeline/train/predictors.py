
def build_sequence_predictors(in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols):
    return [
        "task_encoded",
        
        *in_progress_cols,
        *in_queue_cols,
        *exists_cols,
        *in_throughput_cols,
        
        
        "created_day_of_month",
        "created_week_of_year",
        "created_month_of_year",
        "created_year",

    ]
def build_duration_class_predictors(in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols):
    return [
        "task_encoded",
        
        *in_progress_cols,
        *in_queue_cols,
        *exists_cols,
        *in_throughput_cols,
        
        "created_day_of_month",
        "created_week_of_year",
        "created_month_of_year",
        "created_year",
        
        #From sequence prediction
        "predicted_start_sequence",
        "predicted_complete_sequence",
    ]


def build_duration_regression_predictors(in_progress_cols, in_queue_cols, exists_cols, in_throughput_cols):
    return [
        "task_encoded",

        *in_progress_cols,
        *in_queue_cols,
        *exists_cols,
        *in_throughput_cols,

        "created_day_of_month",
        "created_week_of_year",
        "created_month_of_year",
        "created_year",

        "predicted_start_sequence",
        "predicted_complete_sequence",

        # From duration class prediction
        "predicted_duration_class",
    ]
