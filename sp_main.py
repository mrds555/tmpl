
# # Kfold v3
import pandas as pd
import numpy as np
import optuna
import os
import argparse
from pathlib import Path
import sys

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
# from catboost import CatBoostRegressor, Pool
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from lightgbm import LGBMClassifier

import xgboost as xgb

if os.getenv('CID') == 'tp2250':
    os.environ['KG'] = r'D:\workspace\sync_share\kg'

import importlib
lib_path1 = Path(os.getenv('KG')) / 'lib'
lib_path2 = Path(os.getenv('WS')) / 'sync_share' / 'python' / 'lib'
sys.path.append(str(lib_path1))
sys.path.append(str(lib_path2))

import lib_jk as jk
import kg_jk as kg
import sp_data as spd
import optuna_jk as ojk
import model_jk as mjk
import params as pr
jk.libs()
jk.reload_lib(pr)

import warnings
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)

def main():
        
    parser = argparse.ArgumentParser(description="Time Series Model Trainer")

    parser.add_argument("--device", type=str, default="windows", choices=["windows", "linux"])
    parser.add_argument("--sample_rate", type=float, default=100)
    parser.add_argument("--features", type=str, default="full", choices=["full", "weight", "high", "low", "other"], help="full or other") ## score

    parser.add_argument("--model_type", type=str, default="full", 
                        choices=["full"])  #### must set ## score

    parser.add_argument("--y_transform", type=str, default="False", choices=["True","False"]) #### must set; score; True for low
    parser.add_argument("--w_transform", type=str, default="False", choices=["True","False"]) #### must set; score; True for both
    parser.add_argument("--no_weight", type=str, default="False", choices=["True","False"]) #### must set; score

    parser.add_argument("--code_encoding", type=str, default="False", choices=["True","False"]) ## score
    parser.add_argument("--tuning", type=str, default="reg", choices=["reg","test","score","review"]) ## score

    parser.add_argument("--project", type=str, default="sp")
    parser.add_argument("--study", type=str, default="cat_0504_feats2") # last low weight study was: ts_low_seg_y_w_transform_all_hrz ## score
    parser.add_argument("--n_fold", type=int, default=10) ## score
    parser.add_argument("--trainer", type=str, default="cat")
    parser.add_argument("--ss_model_saved_path", type=str, default="kg/sp_obs/models")
    parser.add_argument("--n_trials", type=int, default=100) ## score
    parser.add_argument("--selector", type=str, default="False") ## score


    INB_MODE = False


    if INB_MODE:
        args, unknown = parser.parse_known_args(args=[]) 
    # elif unknown and INB_MODE is False:
    elif INB_MODE is False:
        # Use the parser's own error method for a consistent look
        args, unknown = parser.parse_known_args()
        if unknown:
            parser.error(f"Unknown arguments detected: {unknown}")
        

    CONFIG = {  "device" : getattr(args,"device"),
                "sample_rate" : getattr(args,"sample_rate"),
                "project" : getattr(args,"project"),
                "features" : getattr(args,"features"), # "full": full list of features - drop_list0
                "study" : getattr(args,"study"),
                "model_type" : getattr(args,"model_type"),
                "y_transform" : getattr(args,"y_transform"),
                "w_transform" : getattr(args,"w_transform"),
                "no_weight" : getattr(args,"no_weight"),
                "code_encoding" : getattr(args,"code_encoding"),
                "n_fold" : getattr(args,"n_fold"),
                "tuning" : getattr(args,"tuning"),
                "trainer" : getattr(args,"trainer"),
                "ss_model_saved_path" : getattr(args,"ss_model_saved_path"),
                "n_trials" : getattr(args,"n_trials"),
                "selector" : getattr(args,"selector"),
            }

    ss_model_saved_path = CONFIG['ss_model_saved_path']
    ss_model_saved_path = Path(ss_model_saved_path)
    n_fold = CONFIG['n_fold']
    trainer_id = CONFIG['trainer']
    n_trials = CONFIG['n_trials']
    selector = CONFIG['selector']


    if any(n in trainer_id for n in ['cat', 'cb']):
        trainer = CatBoostClassifier
    elif any(n in trainer_id for n in ['xg']):
        trainer = xgb.XGBClassifier
    elif any(n in trainer_id for n in ['lgb', 'light']):
        trainer = LGBMClassifier

    # ======================
    # 1. Load Data
    # ======================


    KG = os.getenv('KG')
    train_csv = Path(KG) / 'sp_obs/train.csv'
    train_df = pd.read_csv(str(train_csv))
    train_df.columns = train_df.columns.str.replace('-', '_')

    all_df_cols = train_df.columns
    non_feat_cols = ['ID','y']
    base_feat_cols = list(set(all_df_cols) - set(non_feat_cols))

    X = train_df.drop(columns=non_feat_cols)
    X = spd.add_eng_features(X)

    y = train_df['y']
    y = LabelEncoder().fit_transform(y)
    y = pd.Series(y)


    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Found {len(duplicate_cols)} duplicates: {duplicate_cols}")
        
        # Keep only the first occurrence of each column name
        X = X.loc[:, ~X.columns.duplicated(keep='last')] # without keep='last' leaves first in.
        print(f"Removed duplicates leaving last column only")

    X_shape = X.shape
        
    print(f"X.shape: {X_shape}")

    # ======================
    # 2. Define Objective
    # ======================


    def make_objective(model_search, **config):
        def objective(trial):
        
            model_params = pr.opt_params(trial, trainer_id)
            ms_params = config
            ms_params.update({'trial' : trial})
            if ms_params['selector'] == "True":
                ms_params['feats_selection'] = dict()
            
            outcome, models = model_search(model_params=model_params, ms_params=ms_params)


            ms_params.get('metrics_sets', None)
            opt_direction = opt_direction = list(metrics_sets.values())[0]["direction"]
            primary_metric_key = list(metrics_sets.keys())[0]

            past_values = [t.value for t in study.get_trials(states=[optuna.trial.TrialState.COMPLETE])]
            score = trial.user_attrs.get(primary_metric_key)
            ss_path = trial.user_attrs.get('ss_model_saved_path')
            
            if len(past_values) >= 10:
                
                is_max = 'max' in opt_direction
                pct = 90 if is_max else 10
                threshold = np.percentile(past_values, pct)
            
                # 2. Define the 'Success' condition
                is_top_tier = (score >= threshold) if is_max else (score <= threshold)

                if is_top_tier:
                    print(f"Trial {trial.number} hit the top 10% ({score:.4f}). Saving...")
                    
                    mjk.save_models(trial, models, score, ss_path)
        
            # 3. Single execution block
            if len(past_values) < 10 :
                mjk.save_models(trial, models, score, ss_path)
            
            trial.set_user_attr('seed', config['seeds'])

            excl_p = ['ver']
            param_list = [p for p in model_params.keys() if not any(w in p for w in excl_p)]
            for p in param_list:
                trial.set_user_attr(p, model_params[p])
            
            

            save_params = model_params.copy()

            trial.set_user_attr("params", save_params)
            trial.set_user_attr('x_columns', list(x_columns))
            trial.set_user_attr('n_x_columns', n_x_columns)
            trial.set_user_attr('CID', os.environ['CID'])

            
            
            
            return outcome
        return objective

    # ======================
    # 3. Run Bayesian Optimization
    # ======================

    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    drop_cols = []
    metrics_sets = {
            "f1_score" : {
                "direction": "maximize",
                "vars": ["mean"], # mean, std
                "function" : f1_score,
                "extra_function_params" : {"average" : "macro"}
            },
    }

    # CatBoostClassifier, xgb.XGBClassifier
    ms_params = {
        "trainer" : trainer,
        "X" : X,
        "y" : y,
        "splitter" : skf.split,
        "seeds" : [42],
        "drop_cols" : [],
        "metrics_sets" : metrics_sets,
        "base_feat_cols" : base_feat_cols,
        "non_feat_cols" : non_feat_cols,
        "ss_model_saved_path" : ss_model_saved_path,
        "selector" : selector,
    }


    # config_objective = {
    #     "model_params" : model_params,
    #     "ms_params" : ms_params,
    # }


    x_columns = X.columns
    n_x_columns = len(x_columns)



    opt_direction = list(metrics_sets.values())[0]["direction"]
    conn_string = ojk.optuna_conn()

    study = optuna.create_study(
        direction=opt_direction,
        study_name=CONFIG["study"],
        storage=conn_string,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,  # Let 10 trials finish 5 folds NO MATTER WHAT
            n_warmup_steps=20    # Don't prune any trial until iteration 100
        ),
        sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True, warn_independent_sampling=False),
    )

    study.set_user_attr("model_config", CONFIG)
    study.set_user_attr("X_shape", X_shape)

    # start_params = {'depth': 4, 'l2_leaf_reg': 0.017045438841918885, 'bagging_temperature': 0.7000000000000001, 'random_strength': 0.47932346965289585, 'min_data_in_leaf': 31,}
    # start_params = {'max_depth': 3, 'min_child_weight': 6.0, 'subsample': 1.0, 'colsample_bytree': 0.9, 'gamma': 1.25, 'reg_lambda': 4.0}
    # {'max_depth': 3, 'min_child_weight': 53.21977488232859, 'colsample_bytree': 0.7, 'gamma': 0.33740838786773464, 'reg_alpha': 0.8819964881690022, 'reg_lambda': 4.0067642243873145}
    # {'depth': 2, 'l2_leaf_reg': 0.7967909749578835, 'bagging_temperature': 0.2, 'random_strength': 0.9895751834836317}
    # actual trees =  1051, 759, 764, 831, 878, 439, 1142, 467, 940, 1872

    # study.enqueue_trial(start_params) 

    mjk.model_search.calls = 0

    try:
        study.optimize(make_objective(mjk.model_search, **ms_params), n_trials=n_trials)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Concluding optimize...")

    # best_models = Models[study.best_trial.number]
    # ======================
    # 4. Results
    # ======================

    return study

if __name__ == "__main__":
    study = main()
    print(f"Best value: {study.best_value}")
