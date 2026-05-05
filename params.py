import numpy as np

def opt_params(trial, trainer_id):

    if any(c in trainer_id for c in ['cat', 'cb']):
        model_params = {
                "n_estimators": 3000, # iterations
                "learning_rate": 0.01,
                "depth": trial.suggest_int("depth", 2, 4),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 100, log=True), # 1/lr to 3/lr; too high if L2*LR > 1.0
                # "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1.5, step=0.1),
                # "random_strength": trial.suggest_float("random_strength", 0, 1),
                "min_data_in_leaf" :  trial.suggest_int("min_data_in_leaf", 5, 100),
                # "rsm" :  trial.suggest_float("rsm", 0.5, 1), # col_sampling
                "loss_function": "MultiClass",
                "eval_metric": "MultiClass",
                "random_state": 42,
                "early_stopping_rounds": 50,
                "verbose": 100,
                "task_type": "CPU",  # or "GPU" if you have GPU
                "thread_count" : 2,
                # "task_type": "GPU",  # or "GPU" if you have GPU
                # "gpu_ram_part" : 0.7,
                # "max_ctr_complexity" : 1,
                #  "gpu_cat_features_storage" : "CpuPinnedMemory",
                "use_best_model": True
            }
# {'max_depth': 3, 'min_child_weight': 4.0, 'subsample': 0.9, 'colsample_bytree': 1.0, 'gamma': 1.75, 'reg_lambda': 6.5}
# {'max_depth': 3, 'min_child_weight': 14.0, 'gamma': 0.75, 'reg_lambda': 8.0} actual trees =  452, 757, 343, 753, 447, 355, 868, 426, 732, 578 Best F1: 0.6528071040155728
    elif any(n in trainer_id for n in ['xg']):
        model_params = {
            "objective": "multi:softprob",
            "num_class": 3, # len(np.unique(y)),
            "n_estimators": 3000,
            "learning_rate": 0.01, # trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_child_weight": trial.suggest_int("min_child_weight_int", 5, 100),
            # "subsample": 0.9, # trial.suggest_float("subsample", 0.7, 1.0),
            # "colsample_bytree": 1, # trial.suggest_float("colsample_bytree", 0.7, 1),
            "gamma": trial.suggest_float("gamma", 1e-3, 1, log=True),
            # "reg_alpha": trial.suggest_float("reg_alpha", 1e-1, 1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100, log=True), # G**2 / (H + lambda); if H is small (small leaf), lambda can dominate
            "random_state": 42,
            "tree_method": "exact",
            "eval_metric": "mlogloss",
            "early_stopping_rounds" : 50,
            # "eval_metric": macro_f1,
            # "enable_categorical" : True,
        }
    elif any(c in trainer_id for c in ['lgb', 'light']):
        model_params = {
                "objective": "multiclass", # # "regression_l1"
                "num_class" : 3,
                "metric" : "multi_logloss", # early stopping
                "importance_type" : "gain",
                "n_estimators": 3000,
                "learning_rate": 0.01, #trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 4), # -1, # max_depth,
                "num_leaves" : trial.suggest_int("num_leaves", 7, 31), # .suggest_int("num_leaves", 20, 2**(max_depth+2)) .suggest_categorical("num_leaves", [15, 31, 63, 127]),
                "min_child_samples" : trial.suggest_int("min_child_samples", 5, 100), # min_data_leaves
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 1e+2, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 1e+1, log=True), #step=5),
                # "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-2, 1e+2, log=True), # for y_target only 1 was fine.
                # "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1),
                # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.8, 1), # subsample
                # "bagging_freq": 1, # trial.suggest_int("bagging_freq", 2, 8, step=1), # every k iteration                
                "min_split_gain": 0.001, # trial.suggest_categorical("min_split_gain", [0, 1e-7, 1e-6, 1e-5]), #1e-6,
                "max_bin" : 63,
                "grow_policy": "SymmetricTree", # Usually best for small data
                "n_jobs" : -1,
                "deterministic" : True,
                "force_column_wise" : True,
                "verbosity": -1, 
                # "device" : "gpu",
                # "eval_metric": "mae",
                # 'deterministic' : True,
                # 'force_col_wise' : True,
                # 'force_row_wise' : True,
                # "extra_trees": True, # trial.suggest_categorical("extra_trees", [True, False]),
                # "boosting_type": "dart",           # Replaces "gbdt"
                # "data_sample_strategy": "goss",    # Good for large, noisy 5.3M row data
                # "min_data_per_group": 50000,       # Prevents shredding your 20% categories
                # "cat_smooth": 100,                 # Smooths out categorical noise
                # "drop_rate" : 0.1,
                # "skip_drop" : 0.5,
                # "top_rate" : 0.2,
                # "other_rate" : 0.3,
                }
    else:
        return None
        
    return model_params