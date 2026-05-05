
import lib_jk as jk, libdf_jk as dfjk, libplot_jk as pltjk
import pandas as pd
import numpy as np
import os
import sys
import xgboost as xgb
import optuna
import re
import time
import gc
import psutil
import ipynbname
from pathlib import Path
from IPython.display import display
import math
import matplotlib.pyplot as plt



from optuna.visualization import plot_slice
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostRegressor, CatBoostClassifier,Pool
import lightgbm as lgb
from lightgbm import LGBMRegressor

from optuna.integration import LightGBMPruningCallback

def optuna_conn():
    import urllib.parse
    
    optunapw = os.getenv('optunapw')
    
    DB_PARAMS = {
        "host": '192.168.1.254',
        "database": 'optuna_db',
        "user": 'optuna_user',
        "password": f'{optunapw}',
        "port": '5432',
    }
    
    conn_string = f"postgresql://{DB_PARAMS['user']}:{urllib.parse.quote_plus(DB_PARAMS['password'])}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"

    return conn_string


def model_search(model_params, ms_params):
             
    results = None
    epsilon = 1e-8
    
    model_search.calls += 1
    scores = {}

    if model_search.calls > 1:
        t_break = 3
        print(f'break for {t_break} seconds')
        time.sleep(t_break)

    default_ss_save_location = Path('kaggle/models')
    trainer = ms_params.get('trainer')
    X = ms_params.get('X')
    y = ms_params.get('y')
    weight = ms_params.get('weight', None)
    splitter = ms_params.get('splitter', None) # has to take X and return (train_idx, val_idx)
    seeds = ms_params.get('seeds', [42])
    drop_cols = ms_params.get('drop_cols', [])
    metrics_sets = ms_params.get('metrics_sets', None)
    trial = ms_params.get('trial', None)
    base_feat_cols = ms_params.get('base_feat_cols', [])
    non_feat_cols = ms_params.get('non_feat_cols', [])
    cat_cols = ms_params.get('cat_cols', None)
    ss_model_saved_path = ms_params.get('ss_model_saved_path', None)
    selector_enabled = ms_params.get('selector', False)
    feats_selector_list = ms_params.get('feats_selection', None)
    if ss_model_saved_path is None:
        print('ss_model_saved_path is None. Models will be saved in default location that may be unrelated to the project.')

    opt_direction = list(metrics_sets.values())[0]["direction"]
    if 'max' in opt_direction:
        best_primary_score = -1e+9
    else:
        best_primary_score = 1e+9
    
    
    primary_metric_key = list(metrics_sets.keys())[0]
    primary_metric_func = metrics_sets[primary_metric_key]['function']
    primary_metric_func_extra_params = metrics_sets[primary_metric_key].get('extra_function_params', {})

    if any(kw in str(trainer) for kw in ['xg']):
        trainer_id = 'xgb'
    elif any(kw in str(trainer) for kw in ['cat','cb']):
        trainer_id = 'cat'
    elif any(kw in str(trainer) for kw in ['lgb','light']):
        trainer_id = 'lgb'
    else:
        trainer_id = 'other'

    keywords = ['max', 'min', 'lambda', 'frac','leav','leaf','alpha','freq', 'bagg', 'weight']
    pwords = []
    for k,v in model_params.items():
        if any(word in k for word in keywords):
            pwords.append(f'"{k}" : {v}')
    
    print('')
    print(f">>> {trial.study.study_name}: {trial.user_attrs.get('n_x_columns')} X.columns <<<")
    print("params = {"+(','.join(pwords))+"}", flush=True)



    
    # individual train session
    def sub_train(trainer, X_tr, X_val, y_tr, y_val, w_tr=None, w_val=None): 
        
        trainer_str = str(trainer).lower()
        evals_result = {} 
        fit_params = {}

        def get_fit_params(trainer_id, X_tr, X_val, y_tr, y_val):
            params = {
                'eval_set': [(X_tr, y_tr), (X_val, y_val)],
            }
            
            if any(n in trainer_id for n in ['lgb', 'light']):
                params['eval_names'] = ['train', 'valid']
            elif any(n in trainer_id for n in ['cat', 'cb']):
                # params['eval_set'] = (X_val, y_val) # <= replaced by two lines below
                if cat_cols is not None:
                    cat_features_idx = [i for i, col in enumerate(X_tr.columns) if col in cat_cols]
                else:
                    cat_features_idx = []
                X_tr = Pool(X_tr, y_tr, cat_features=cat_features_idx)
                # Convert validation data to Pool
                params['eval_set'] = Pool(X_val, y_val, cat_features=cat_features_idx)
            elif any(n in trainer_id for n in ['xg', 'xgb']):
                params['verbose'] = False
            # REMOVED: elif any(n in trainer_id for n in ['xg']): params['evals_result'] = evals_result
            
            return X_tr, y_tr, params # fallback: remove X_tr, y_tr
            
        def get_trainer_params(trainer_id, base_params):
            params = base_params.copy()
            
            if any(n in trainer_id for n in ['xg', 'xgb']):
                params.pop('eval_metric', None)
                params.update({'eval_metric' : 'mlogloss'})
            elif any(n in trainer_id for n in ['cat', 'cb']):
                params.update({'loss_function': 'MultiClass', 'eval_metric': 'MultiClass'})
            return params
        
        # fit_params = get_fit_params(str(trainer), X_tr, X_val, y_tr, y_val) # replaced with below
        trainer_params = get_trainer_params(trainer_str, model_params)

        #### SELECTOR START ####
        if selector_enabled.lower() == 'true':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import SelectFromModel
            from collections import Counter

            # We use a simple version of your trainer to find the 'Lift' 
            # without triggering callbacks or Pools yet.
            from sklearn.feature_selection import SelectFromModel
            
            selector_params = trainer_params.copy()
            selector_params.pop('use_best_model', None)
            selector_params.pop('early_stopping_rounds', None)
            selector_params['n_estimators'] = 100 

            # Use a simpler, faster version of your model for selection
            # Using 'mean' or '1.25*median' to kill the 190 over-engineered noise features
            selector_model = trainer(**selector_params)
            selector = SelectFromModel(selector_model, threshold="1.25*median")
            
            # Fit on the raw DataFrames
            selector.fit(X_tr, y_tr)
            selected_cols = X_tr.columns[selector.get_support()]

            for feat in selected_cols:
                feats_selector_list[feat] = feats_selector_list.get(feat, 0) + 1

        
            # Prune the DataFrames BEFORE they get turned into Pools/DMatrix
            X_tr = X_tr[selected_cols]
            X_val = X_val[selected_cols]
        
            print(f"[{trainer_str}] Selected {len(selected_cols)} features for this fold.")
        #### SELECTOR END ####


        X_tr, y_tr, fit_params = get_fit_params(trainer_str, X_tr, X_val, y_tr, y_val)
        

        if trainer_str.find('xgb') != -1 or trainer_str.find('cat') != -1:

            model = trainer(**trainer_params)

            # local_params = get_fit_params('xgb', X_tr, X_val, y_tr, y_val)

        else:
            # log_cb = lgb.log_evaluation(period=0)
            # log_cb = VariablePeriodLogger(start_limit=10, final_period=100) 
            # Initialize the custom logger: 1-10 individually, then 50, 100, 150...
            log_cb = StepLogger(initial_limit=10, period=100)
            # log_cb = lgb.log_evaluation(period=None)

            
            es_cb = lgb.early_stopping(stopping_rounds = 100, verbose = False)
            re_cb = lgb.record_evaluation(evals_result)

            callbacks = [es_cb,log_cb, re_cb] #  
            # if trial is not None and fold == 0:
            #     pc_cb = LightGBMPruningCallback(trial, "rmse", valid_name="valid")
            #     callbacks.append(pc_cb)

            fit_params.update({'callbacks' : callbacks})

            model = trainer(**trainer_params)
        
        if isinstance(X_tr, Pool):
            model.fit(X_tr, y=None, **fit_params)
        else:
            model.fit(X_tr, y_tr, **fit_params,
            # eval_set=[(X_val, y_val)],  # <-- validation data
            # verbose=False,
            )

        train_start = tr_idx[0]
        train_end = tr_idx[-1]
        
        # Get first and last values of the test indices
        val_start = val_idx[0]
        val_end = val_idx[-1]

        
        if True: # 'valid' in evals_result:
            # 1. Identify the framework and metric name dynamically
            if hasattr(model, 'get_best_score'):  # CatBoost
                best_score_dict = model.get_best_score()
                # CatBoost structure: {'learn': {...}, 'validation': {'RMSE': 0.123}}
                eval_key = 'validation' if 'validation' in best_score_dict else 'test'
                metric_name = list(best_score_dict[eval_key].keys())[0]
                best_score = best_score_dict[eval_key][metric_name]
                best_iteration = model.get_best_iteration()
                evals = model.get_evals_result()
                tr_loss_rdc = evals['learn'][metric_name][0] - evals['learn'][metric_name][-1]
                val_loss_rdc = evals[eval_key][metric_name][0] - evals[eval_key][metric_name][-1]
            
            elif hasattr(model, 'best_score_'):  # LightGBM
                # LightGBM structure: {'valid_1': {'rmse': 0.123}}
                eval_key = list(model.best_score_.keys())[0]
                metric_name = list(model.best_score_[eval_key].keys())[0]
                best_score = model.best_score_[eval_key][metric_name]
                best_iteration = model.best_iteration_
                evals = model.evals_result_
                tr_loss_rdc = evals['train'][metric_name][0] - evals['train'][metric_name][-1]
                val_loss_rdc = evals['valid'][metric_name][0] - evals['valid'][metric_name][-1]
            
            else:  # XGBoost
                # XGBoost uses .best_score (no underscore)
                best_score = model.best_score
                best_iteration = model.best_iteration
                # XGBoost doesn't store the metric name in best_score, 
                # but we can find it in eval_results
                evals = model.evals_result() 
                if evals:
                    eval_key = list(evals.keys())[0]        # e.g., 'validation_0'
                    metric_name = list(evals[eval_key].keys())[0] # e.g., 'mlogloss'
                else:
                    eval_key, metric_name = None, None
                tr_loss_rdc = evals['validation_0'][metric_name][0] - evals['validation_0'][metric_name][-1]
                val_loss_rdc = evals['validation_1'][metric_name][0] - evals['validation_1'][metric_name][-1]

            val_tr_ratio = val_loss_rdc / (tr_loss_rdc + epsilon)

            print(f'\nFold {fold} summary:')
            if evals:
                # 1. Get the first validation set name (e.g., 'valid_0', 'validation', or 'learn')
                first_set = list(evals.keys())[0]
                # 2. Get the list of all metrics in that set
                metric_names = list(evals[first_set].keys())
                # 3. Grab the primary metric (usually the first one)
                primary_metric = metric_names[0]

                #inclusive [starrt:end]
                train_size = X_tr.num_row() if hasattr(X_tr, 'num_row') else len(X_tr)
                val_size = X_val.num_row() if hasattr(X_val, 'num_row') else len(X_val)
                
                # print(f'train_size: {len(X_tr)} [{train_start}:{train_end}], val_size: {len(X_val)} [{val_start}:{val_end}]')
                print(f' - train_size: {train_size} [{train_start}:{train_end}], val_size: {val_size} [{val_start}:{val_end}]')
                print(f' - val-to-train loss reduction ratio: {val_tr_ratio:.1%}')

                # print(f'(best_iter_trees = {model.best_iteration_}, tried_trees = {model.booster_.num_trees()})', flush=True)
        
                # tree_info = model.booster_.dump_model()['tree_info']
                # leaves = [t['num_leaves'] for t in tree_info]
                # print(f"trees={len(tree_info)}, ", end='')
                # print(f"avg leaves={sum(leaves)/len(leaves):.1f}, ", end='')
                # print(f"max leaves={max(leaves)}")
    
            
            # 2. Universal Print and Append
            print(f" - Metric Used:    {metric_name}")
            print(f" - True Best Score: {best_score:.8f}")
            print(f" - At Iteration:   {best_iteration}")
            
            best_val_scores.append(best_score)
            tr_loss_rdcs.append(tr_loss_rdc)
            val_loss_rdcs.append(val_loss_rdc)
            val_tr_ratios.append(val_tr_ratio)

            
        return model

    for i_seed, seed in enumerate(seeds):

        models, trees, fold_unused_feats, n_fold_unused_feats, best_val_scores, fold_scores = [], [], [], [], [], []
        tr_loss_rdcs, val_loss_rdcs, val_tr_ratio, val_tr_ratios = [], [], [], []
        
        oof = np.zeros(len(X))
        mask_oof = np.zeros(len(X), dtype=bool)
        
        for fold, (tr_idx, val_idx) in enumerate(splitter(X, y)):
            X_tr, X_val = X.iloc[tr_idx].drop(columns=drop_cols, errors='ignore'), X.iloc[val_idx].drop(columns=drop_cols, errors='ignore')
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            mask_oof[val_idx] = True # Mark these rows as "predicted"
             
            model = sub_train(trainer, X_tr, X_val, y_tr, y_val)   


            
            if str(trainer).lower().find('cat') != -1:
                oof[val_idx] = np.squeeze(model.predict(X_val))
                
            else:
                oof[val_idx] = model.predict(X_val)

            fold_preds = oof[val_idx]

            corr = np.corrcoef(y_val.values.flatten(), oof[val_idx].flatten())[0, 1]

            fold_score = primary_metric_func(y_val, fold_preds, **primary_metric_func_extra_params)

            fold_scores.append(fold_score)
            
            print(f"DEBUG Fold {fold}:")
            print(f"  - Validation Index Range: {val_idx[0]} to {val_idx[-1]}, (size={len(val_idx)})")
            print(f"  - Pred vs Target Corr: {corr:.4f}")
            print(f"  - Fold-specific Score: {fold_score:.6f}")
            print(f"  - Pred Mean: {fold_preds.mean():.6f} | Target Mean: {y_val.mean():.6f}")
                
            # Universal best iteration extraction
            if hasattr(model, 'get_best_iteration'):
                # CatBoost method (returns None if no eval_set)
                best_iter = model.get_best_iteration() or model.tree_count_
            elif hasattr(model, 'best_iteration'):
                # XGBoost attribute
                best_iter = model.best_iteration
            elif hasattr(model, 'best_iteration_'):
                # LightGBM / CatBoost attribute
                best_iter = model.best_iteration_
            else:
                # Fallback to total trees built
                best_iter = getattr(model, 'tree_count_', getattr(model, 'n_estimators', 0))
                # best_iter = model.n_estimators
            
            models.append(model)
            trees.append(best_iter)

            # 2. Handle Feature Names (Universal list of strings)
            if hasattr(model, 'feature_names_in_'): # Scikit-Learn standard (Works for newer XGBoost & CatBoost)
                feat_names = [str(c) for c in model.feature_names_in_]
            elif hasattr(model, 'feature_name_'): # LightGBM
                feat_names = [str(c) for c in model.feature_name_]
            elif hasattr(model, 'feature_names_'): # CatBoost
                feat_names = [str(c) for c in model.feature_names_]
            else:
                try:
                    feat_names = model.get_booster().feature_names
                except:
                    feat_names = []
            
            # 3. Handle Feature Importance
            if hasattr(model, 'feature_importances_'):
                # Standard for Sklearn wrappers (XGB/Cat/LGBM)
                importances = model.feature_importances_
            elif hasattr(model, 'get_booster'):
                # Native XGB
                importances = list(model.get_booster().get_score(importance_type='weight').values())
            else:
                importances = []
            
            # 4. Calculate Unused Features
            if len(feat_names) == len(importances):
                unused_feats = [name for name, imp in zip(feat_names, importances) if imp == 0]
                fold_unused_feats.append(unused_feats)
                n_fold_unused_feats.append(len(unused_feats))


            # results = model.evals_result_
            

            
            if fold == 0:
                # # 1. Framework-agnostic feature name extraction
                # if hasattr(model, 'feature_names_in_'): # Scikit-Learn standard (Works for newer XGBoost & CatBoost)
                #     mc = list(model.feature_names_in_)
                # elif hasattr(model, 'feature_name_'): # LightGBM specific
                #     mc = model.feature_name_
                # elif hasattr(model, 'feature_names_'): # CatBoost specific
                #     mc = model.feature_names_
                # else:
                #     # Native XGBoost / Fallback
                #     try:
                #         mc = model.get_booster().feature_names
                #     except:
                #         mc = []
                
                # 2. Logic remains the same
                base_feat = [str(c) for c in feat_names if c in base_feat_cols]
                non_feat_feat = [str(c) for c in feat_names if c in non_feat_cols]
                ext_base_feat = [str(c) for c in feat_names if c not in base_feat_cols + non_feat_cols]
    
                # 3. Print and Save
                print(f'Fold {fold} note:')
                print(f' - {len(base_feat)} base features')
                print(f' - {len(non_feat_feat)} non-"feature" names : {non_feat_feat}')
                print(f' - {len(ext_base_feat)} extra base features: {ext_base_feat}')
                print(f' - {len(feat_names)} total features in model wrapper', flush=True)
                
                # 3. Store results
                # mc = [str(c) for c in mc]
                trial.set_user_attr('n_features_model_wrapper', len(feat_names)) 
                trial.set_user_attr('actual_features_model_wrapper', feat_names)

            # print(f'(rmse) train: {results['train']['rmse'][-1]:.4f}, val: {results['valid']['rmse'][-1]:.4f} ', end='')
            # # Get first and last values of the training indices

            # #inclusive [starrt:end]
            # print(f'train_size: {len(X_tr)} [{train_start}:{train_end}], val_size: {len(X_val)} [{val_start}:{val_end}]')
            # print(f'(best_iter_trees = {model.best_iteration_}, tried_trees = {model.booster_.num_trees()})', flush=True)
    
            # tree_info = model.booster_.dump_model()['tree_info']
            # leaves = [t['num_leaves'] for t in tree_info]
            # print(f"trees={len(tree_info)}, ", end='')
            # print(f"avg leaves={sum(leaves)/len(leaves):.1f}, ", end='')
            # print(f"max leaves={max(leaves)}")




        #################### folds concluded ####################    

        

        oof_val = oof[mask_oof]
        y_target_val = y.iloc[mask_oof]

        for i, (k,v) in enumerate(metrics_sets.items()):
            score_func = metrics_sets[k]['function']
            extra_params = metrics_sets[k].get('extra_function_params', {})
            # y_target_val = np.array(y_target_val)
            # oof_val = np.array(oof_val)
            score = score_func(y_target_val, oof_val,**extra_params)
            if i == 0:
                primary_score = score
            scores.update ({k : score})
            trial.set_user_attr(f'{k}' , score) 

        # using last model?
        # bst = model.booster_
        # feat_names = bst.feature_name()
        # unused_feats = [name for name, imp in zip(feat_names, importances) if imp == 0]
        # fold_unused_feats.append(unused_feats) # (",".join(unused_feats))
        # n_fold_unused_feats.append(len(unused_feats))

        # base_path = Path(__file__).parent.resolve()
        # model_dir = base_path / "models"


        # base_path = Path(__file__).parent.resolve()
        # model_dir = base_path / "models"
        if ss_model_saved_path is not None:
            ss_path = ss_model_saved_path
        else:
            ss_path = default_ss_save_location
        trial.set_user_attr('ss_model_saved_path', ss_path.as_posix())
        trial.set_user_attr('trees', trees)

        # save best models and stuff
    
    ################################ seed loop ends ######################################## 
    
    trial.set_user_attr('trainer_id', trainer_id)
    trial.set_user_attr('trainer_best_val_scores', np.round(best_val_scores, 4).tolist())
    trial.set_user_attr('train_loss_reduced', np.round(tr_loss_rdcs, 4).tolist())
    trial.set_user_attr('val_loss_reduced', np.round(val_loss_rdcs, 4).tolist())
    trial.set_user_attr('val_train_loss_ratios', np.round(val_tr_ratios, 4).tolist() )
    val_tr_ratios_mean = np.mean(val_tr_ratios)
    val_tr_ratios_std = np.std(val_tr_ratios)
    trial.set_user_attr('val_train_loss_ratio_avg', np.mean(val_tr_ratios_mean))
    trial.set_user_attr('val_train_loss_ratio_std', np.mean(val_tr_ratios_std))
    trial.set_user_attr('primary_metric_fold_scores', np.round(fold_scores, 4).tolist())
    trial.set_user_attr('std_primary_metric_fold_scores', np.std(fold_scores))
    trial.set_user_attr("unused_features", fold_unused_feats)
    trial.set_user_attr("n_unused_features", n_fold_unused_feats)
    if selector_enabled.lower() == "true":
        trial.set_user_attr("feats_selector_list", feats_selector_list)

    score = scores[list(scores.items())[0][0]]

    for (k,v) in metrics_sets.items():
        pass
    
    print(f'Trial {trial.number} summary for study: {trial.study.study_name} (local session # {model_search.calls}):')
    print(f' - val-to-train loss reduction ratio mean: {val_tr_ratios_mean:.1%}')
    print(f' - val-to-train loss reduction ratio std: {val_tr_ratios_std:.1%}')
    print(f' - primary metric score = {primary_score}')
    if model_search.calls > 1:
        print(f' - best study params: {trial.study.best_params} by {trial.study.best_trial.user_attrs.get('CID')}')

    sys.stdout.flush()

    gc.collect()
    
    
    return score, models

def save_models(trial, models, score, ss_path):

    model_filenames = []
    
    SS = Path(os.getenv('SS'))
    
    model_dir = SS / ss_path
    model_dir.mkdir(parents=True, exist_ok=True)
    add_tag = jk.get_timestamp()
    formatted_prim_score = f"{score:#.6g}".replace('.', '_')
    for i, model in enumerate(models):
        # 1. Determine the best extension for the specific library
        if hasattr(model, 'get_booster'): # XGB
            ext = ".json"
            bst = model.get_booster()
        elif hasattr(model, 'booster_'): # LGBM
            ext = ".txt" 
            bst = model.booster_
        else:                            # CatBoost
            ext = ".json" # or .cbm for binary
            bst = model
    
        fname = f"{formatted_prim_score}_{trial.study.study_name}_trial_{trial.number}_fold_{i}_{add_tag}{ext}"
        fpath = model_dir / fname
        
        bst.save_model(str(fpath))
        model_filenames.append(fname)
    
    trial.set_user_attr("model_filenames", model_filenames)

class StepLogger:
    def __init__(self, initial_limit=10, period=50):
        self.initial_limit = initial_limit
        self.period = period

    def __call__(self, env):
        # env.iteration is 0-indexed
        count = env.iteration + 1
        
        # Condition: Show every step for 1-10, then every 50, 100, 150...
        if count <= self.initial_limit or count % self.period == 0:
            # Format the output to match standard LGBM logging
            metrics = []
            for data_name, eval_name, val, _ in env.evaluation_result_list:
                metrics.append(f"{data_name}'s {eval_name}: {val:.6f}")
            
            print(f"[{count}]\t" + "\t".join(metrics), flush=True)




import polars as pl
import numpy as np


def apply_df_str_mask(df_str_mask, df_str_name, target_df):
    # This tells the evaluator: "whenever you see 'df', use target_df"
    return pd.eval(df_str_mask, local_dict={df_str_name: target_df}, engine='python')

# --- Usage ---
# str_mask = "(df['weight'] > 1e+06) & (df['y_target'] > -1e-3) & (df['y_target'] <= -1e-4)"
# Apply it to X, or any other dataframe name
# mask = apply_string_mask(X, str_mask)
# X['hw_ne-3_ne-4'] = mask

# same as above; should be interchangeable
def apply_df_str_mask_eval(df_str_mask, str_df_name, target_df):
    target_df_name = "_very_unique_name"
    # 1. Swap 'df' for the new name
    new_str = df_str_mask.replace(str_df_name, target_df_name)
    
    # 2. Eval using the local context where target_name exists
    return eval(new_str, {target_df_name: target_df})
# Usage
# mask = replace_and_eval("X", X, str_mask)

    
def deep_int(X, df_str_mask, bin_col, sample_ratio=3, excl=[], incl=[], depth=4, decimals=4):
    import pandas as pd
    import numpy as np

    X[bin_col] = apply_df_str_mask(df_str_mask, "uniqued_f", X)
    pop_size = len(X)
    BeamX = X[X[bin_col]==True] # only BeamX
    sub_size = len(BeamX)
    max_ratio = np.floor((pop_size / sub_size - 1) * 100) / 100
    print(f'Note BeamX size = {len(BeamX)}. Max sample ratio = {max_ratio:.2f}. Entered sample ratio = {sample_ratio}')
    if pop_size / sub_size - 1 < sample_ratio: 
        print(f'Max exceeded. Using max ratio {max_ratio} instead.')
        sample_ratio = max_ratio
        
        
    BeamX = pd.concat([BeamX, X[X[bin_col]==False].sample(int(len(BeamX)*sample_ratio), random_state=42).copy()], axis=0)
    
    if len(incl) > 0:
        cols = incl
    else:
        cols = [c for c in X.columns if c not in (excl+[bin_col])]
    
    # 1. Identify your column types
    print("here")
    cat_cols = BeamX[cols].select_dtypes(include=['category', 'object']).columns
    print("here2")
    num_cols = BeamX[cols].select_dtypes(exclude=['category', 'object']).columns
    
    # 2. Handle Categoricals: Add 'MISSING' to the allowed categories first
    for col in cat_cols:
        print(col)
        if "MISSING" not in BeamX[col].cat.categories:
            BeamX[col] = BeamX[col].cat.add_categories("MISSING")
        BeamX[col] = BeamX[col].fillna("MISSING")
    
    # 3. Handle Numericals: Fill with 0 (Standard for Lags/Weights)
    BeamX[num_cols] = BeamX[num_cols].fillna(0)
    
    # 4. Final Validation: Ensure NO NaNs remain
    print("Remaining NaNs:", BeamX[cols].isna().sum().sum())
    
    X_rules = BeamX[cols].copy()
    
    # Convert all 'category' columns to integer codes (0, 1, 2...)
    for col in cat_cols:
        X_rules[col] = X_rules[col].cat.codes
    
    # Define your target for 'Superweights'
    y_target = (BeamX[bin_col]==True).astype(int)

    from sklearn.tree import DecisionTreeClassifier, export_text
    
    # Train a shallow tree to find the broad rules
    # 'balanced' helps because weights > 1e08 are likely rare
    dt = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42)
    dt.fit(X_rules, y_target)
    
    # Print the "Reverse Engineered" conditions
    rules_text = export_text(dt, feature_names=list(X_rules.columns), decimals=decimals)
    print(f"--- Rules for Weight = {df_str_mask} ---")
    print(rules_text)
    return len(BeamX[cols[0]])
        

def str_to_booleans(df, rules_text, df_str_mask, any_column='weight'):
    # Filter out empty lines
    
    lines = [line for line in rules_text.split('\n') if line.strip()]
    path_stack = {} 
    paths = []
    ext_paths = []
    classes = [] # New list to store class labels
    
    df_name = [name for name, val in globals().items() if val is df]
    if len(df_name) > 0:
        df_name = [c for c in df_name if c != "df"][0]
    else:
        df_name = "X"
    
    for i, line in enumerate(lines):
        # 1. Count indentation depth based on the number of '|'
        depth = line.count('|')
        
        # 2. Clean the condition (remove symbols)
        clean_rule = line.replace('|', '').replace('---', '').strip()
        
        # 3. Determine if this line is the end of a branch (a leaf)
        is_leaf = False
        current_class = "Unknown" # Default
        
        if "class:" in clean_rule:
            is_leaf = True
            current_class = clean_rule.split("class:")[1].strip()
        elif i + 1 < len(lines):
            next_line = lines[i+1]
            next_depth = lines[i+1].count('|')
            # If the next line moves back or stays same, current is a leaf
            if next_depth <= depth:
                is_leaf = True
                # If current isn't a class line, look ahead for a sibling/child class
                if "class:" in next_line and next_depth == depth:
                     current_class = next_line.replace('|','').replace('---','').split("class:")[1].strip()
        else:
            is_leaf = True # The last line of the tree is always a leaf

        # 4. Update the path stack with the current rule (unless it's just a class label)
        if "class:" not in clean_rule:
            path_stack[depth] = clean_rule
            # Important: Remove any deeper stale paths from previous branches
            path_stack = {k: v for k, v in path_stack.items() if k <= depth}
        
        # 5. If it's a leaf, compile the full path logic
        if is_leaf:
            # 1. Filter out any "class:" labels from the conditions stack
            conditions = [path_stack[d] for d in sorted(path_stack.keys()) 
                          if d <= depth and "class:" not in path_stack[d]]
            ext_logic_parts = []
            logic_parts = []
            
            for c in conditions:
                if c.startswith('~'):
                    clean_c = c[1:].strip() # Remove ~ and extra space
                    ext_logic_parts.append(f"(~{df_name}.{clean_c})")
                    logic_parts.append(f"(~df.{clean_c})")
                else:
                    ext_logic_parts.append(f"({df_name}.{c})")
                    logic_parts.append(f"(df.{c})")
            
            ext_logic = " & ".join(ext_logic_parts)
            logic = " & ".join(logic_parts)
            
            ext_paths.append(f"({ext_logic})")
            paths.append(f"({logic})")
            classes.append(current_class) # Store the class for this path
            
            
            # conditions = [path_stack[d] for d in sorted(path_stack.keys()) if d <= depth]
            # ext_logic = " & ".join([f"({df_name}.{c})" for c in conditions])
            # logic = " & ".join([f"(df.{c})" for c in conditions])
            # ext_paths.append(f"({ext_logic})")
            # paths.append(f"({logic})")

    # This will print the logic for EVERY terminal leaf in the tree

    print(f'target conditions = {df_str_mask.replace("uniqued_f", df_name)}')
    pop_matched = apply_df_str_mask(df_str_mask, "uniqued_f", target_df=df)
    pop = len(df)
    pop_rate = (len(pop_matched[pop_matched==True]) / pop) if pop != 0 else 0
    print(f'target % in population  = {pop_rate*100:.1f}% - {pop_rate * pop:,.0f}')
    
    for i, p in enumerate(ext_paths):
        print(f"Path {i}: {p} -> Class: {classes[i]}")
        
    
        # 2. Pick a rule (e.g., the first one) and evaluate it
        # eval() turns the string into the actual Boolean Series
        rule = eval(paths[i])
        
        # 3. Your existing coverage/precision logic
        coverage = rule.mean()
        # precision = df.loc[rule, 'weight'].gt(high_wgt_threshold).mean() # rule is selection rule that isn't the same as target condition
        selection = df.loc[rule]
        matched = apply_df_str_mask(df_str_mask, "uniqued_f", target_df=selection)
        
        
        called = len(df.loc[rule,any_column])
        called = len(selection)
        selection_pop = coverage * pop 
        precision = len(matched[matched==True]) / called if called != 0 else 0
        target_pop = pop_rate * pop
        matched = precision * called
        recall = matched / target_pop

        print(f"Coverage: {coverage:.1%} (n={selection_pop:.0f} / {pop}), ", end='')
        print(f"Precision: {precision:.1%} (n={matched:.0f} / {called}), ", end='')
        print(f"Recall: {recall:.1%} (n={matched} / {target_pop:.0f})")
    
    # return results
extract_logic_paths = str_to_booleans


def compare_md_features(model, df): # gold
    in_df = df.columns


    in_model = get_features(model)
    # if hasattr(model, 'feature_names_in_'): # Scikit-Learn standard (Works for newer XGBoost & CatBoost)
    #     in_model = [str(c) for c in model.feature_names_in_]
    # elif hasattr(model, 'feature_name'): # LightGBM
    #     # in_model = [str(c) for c in model.feature_name_]
    #     in_model = [str(c) for c in model.feature_name()]
    # elif hasattr(model, 'feature_names'): # CatBoost
    #     # in_model = [str(c) for c in model.feature_names_]
    #     in_model = [str(c) for c in model.feature_names()]
    # else:
    #     try:
    #         in_model = model.get_booster().feature_names
    #     except:
    #         in_model = []
    

    
    in_model_not_in_df = set(in_model) - set(in_df)
    not_in_model_in_df = set(in_df) - set(in_model)
    print(f"in model, not in df: {in_model_not_in_df}")
    print(f"not in model, in df: {not_in_model_in_df}")

    return list(not_in_model_in_df)
