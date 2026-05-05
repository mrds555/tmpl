import os

import optuna
from optuna.visualization import plot_slice
from optuna.integration import LightGBMPruningCallback


def optuna_tuning_review(study, params):
    # This creates an X-Y plot for every parameter in your study
    # X = Parameter Value, Y = Objective Value (Loss)
    fig = plot_slice(study, params)
    fig.show()
    
# params=['depth',  'l2_leaf_reg', 'bagging_temperature']
# optuna_tuning_review(study, params)


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

def optuna_study(input=None):
    '''
    input can be string or number
    '''
    
    conn_string = optuna_conn()

    if input is None:
        study_names = optuna.study.get_all_study_names(storage=conn_string)
        study_name = study_names[-1]
    elif isinstance(input, (int, float)):
        study_names = optuna.study.get_all_study_names(storage=conn_string)
        study_name=study_names[input]
    else:
        study_name = input

    print(f'"{study_name}"', end='')
    study = optuna.load_study(
    study_name=study_name,
    storage=conn_string
    )
    print(f'(last trial = {study.trials[-1].number}, best_trial.number: {study.best_trial.number}, best_value: {study.best_value:.6f})')
    print(f'best_params: {study.best_params}')
    print(f'best_trial trees: {study.best_trial.user_attrs["trees"]}')
    print(f'best_user_attrs_params: {study.best_trial.user_attrs.get('params')}')
    print(f'model config: {study.user_attrs["model_config"]}')

    return study

optuna_get_study = optuna_study 
get_study = optuna_study

def optuna_delete_study(study_name):
    try:
        optuna.delete_study(study_name=study_name, storage=optuna_conn())
        print(f"Study {study_name} deleted successfully.")
    except KeyError:
        print("Study not found.")

def update_optuna_study(): #WIP
    study = kg.optuna_study()
    for i in range (163,233):
        try:
            # Try to fetch models for the current trial
            models = kg.get_models(trial=i)
        except KeyError:
            # If 'model_filenames' is missing, skip this trial and move to the next 'i'
            print(f"Skipping Trial {i}: No models found.")
            continue
        for fold in range(len(models)):
            bst = models[fold]   
            feat_names = bst.feature_name()
            importances = bst.feature_importance(importance_type='split')
            unused_feats = [name for name, imp in zip(feat_names, importances) if imp == 0]
            study.trials[i].set_user_attr("unused_features", ",".join(unused_feats))
            study.trials[i].set_user_attr("n_unused", len(unused_feats))
            

def optuna_study_names(study_id=None):
    import os
    import urllib.parse

    optunapw = os.getenv('optunapw')

    DB_PARAMS = {
        "host": '192.168.1.254',
        "database": 'optuna_db',
        "user": 'optuna_user',
        "password": f'{optunapw}',
        "port": '5432' # Default PostgreSQL port
    }
    
    conn_string = f"postgresql://{DB_PARAMS['user']}:{urllib.parse.quote_plus(DB_PARAMS['password'])}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"
    
    study_names = optuna.study.get_all_study_names(storage=conn_string)
    if study_id is None:
        print(study_names)
        return study_names
    else:
        print(study_names[study_id])
        return study_names[study_id]
    
def optuna_summary(ref=None, trial=None):
    import optuna
    import os
    import urllib.parse

    optunapw = os.getenv('optunapw')

    DB_PARAMS = {
        "host": '192.168.1.254',
        "database": 'optuna_db',
        "user": 'optuna_user',
        "password": f'{optunapw}',
        "port": '5432' # Default PostgreSQL port
    }
    
    conn_string = f"postgresql://{DB_PARAMS['user']}:{urllib.parse.quote_plus(DB_PARAMS['password'])}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"

    if isinstance(ref, optuna.study.Study):
        study = ref
        study_name = study.study_name
    else:
        if isinstance(ref, str):
            study_name = ref
        else:
            study_names = optuna.study.get_all_study_names(storage=conn_string)
            # study_summaries = optuna.study.get_all_study_summaries(storage=conn_string)
            # last_study_name = study_summaries[-(latest+1)].study_name
            if ref is None:
                study_name = study_names[-1]
            else:
                study_name = study_names[ref]

        study = optuna.load_study(
        study_name=study_name, 
        storage=conn_string
        )

    last_trial = study.trials[-1] if study.trials else None
    n_last_trial = last_trial.number
    
    if trial is not None:
        models = get_models(study_name, trial)
        
    else:
        models = get_models(study_name)

    for i, bst in enumerate(models):
        tree_info = bst.dump_model()['tree_info']
        actual_max_leaves = max([tree['num_leaves'] for tree in tree_info])
        print(f"Fold {i} - Actual Max Leaves: {actual_max_leaves}")
        
    
    
    print(f'best_value: {study.best_value:.6f}, ', end='')
    print(f'best_trial.number: {study.best_trial.number}')
    print(f'best_params: {study.best_params}')
    print(f'best_trial CID: {study.best_trial.user_attrs["CID"]}')
    print(f'best_trial trees: {(study.best_trial.user_attrs["trees"][0])}')
    
    
    print(f'last trial: {n_last_trial}')
   
    if trial is None:
        print(f'value: {study.trials[-1].value}, ', end='')
        trees = last_trial.user_attrs.get("trees", "N/A") if last_trial else "No Trials"
        print(f'trees #: {trees[0]}')
    else:
        print(f'trial #: {study.trials[trial].number}')
        print(f'value: {study.trials[trial].value}')
        print(f'CID: {study.trials[trial].user_attrs["CID"]}')
        print(f'trees #: {study.trials[trial].user_attrs["trees"]}')
    return study
        


def get_models(study_name=None, trial=None, option=None): # returns booster(s)
    import optuna
    import lightgbm as lgb
    from pathlib import Path
    import urllib.parse
    import os

    SS = os.getenv('SS')
    study = optuna_study(study_name)
    
    last_trial = study.trials[-1] if study.trials else None
    
    # 1. Get the filenames from the best trial
    ss_manual_backup = 'kg/stock_price/models'
    
    if  trial is None or trial=="best":
        trial_number = study.best_trial.number
        print(f'Loading from best trial={trial_number}')
        target_fnames = study.best_trial.user_attrs["model_filenames"]
        SS_path = study.best_trial.user_attrs.get('ss_model_saved_path', ss_manual_backup)
        trainer_id = study.best_trial.user_attrs.get('trainer_id', None)
        
        
    else:
        trial_number = study.trials[trial].number
        print(f'Loading from trial number={trial_number}')
        target_fnames = study.trials[trial].user_attrs["model_filenames"]
        SS_path = study.trials[trial].user_attrs.get('ss_model_saved_path', ss_manual_backup)
        trainer_id = study.trials[trial].user_attrs.get('trainer_id', None)

    
    print(target_fnames)

    
    SS_env = os.getenv('SS', '.')
    rel_path = study.trials[trial_number].user_attrs.get('ss_model_saved_path', ss_manual_backup)
    model_dir = Path(SS_env) / rel_path    
    
    # 2. Reconstruct the 5 Boosters
    models = []
    for i, fname in enumerate(target_fnames):
        fpath = model_dir / fname
        if i < 2: print(fpath)
        if fpath.exists():
            if fpath.suffix == '.txt':
                with open(fpath) as f:
                    models.append(lgb.Booster(model_str=f.read()))
                    # best_models.append(lgb.Booster(model_file=str(fpath)))

            # Add xgboost branch
            elif 'xgb' in fname.lower() or 'xg' in fname.lower() or 'xg' in trainer_id:
                bst = xgb.Booster()
                bst.load_model(str(fpath))
                models.append(bst)
            
            # Add catboost branch
            elif 'cat' in fname.lower() or 'cb' in fname.lower() or fpath.suffix == '.cbm' or 'cat' in trainer_id:
                bst = CatBoostClassifier()
                # Try loading as binary if JSON fails
                bst.load_model(fpath, format='cbm')
                # bst = cb.CatBoostClassifier()
                # bst.load_model(str(fpath))
                models.append(bst)
        
        else:
            print(f"🚨 Warning: File {fname} not found on this machine!")
            print(f"{fpath}")
    
    print(f'tuned params={study.trials[trial_number].params}')
    print(f'user_attrs_params={study.trials[trial_number].user_attrs.get('params')}')
    print(f'value (score)={study.trials[trial_number].value}')
    print(f'trees={study.trials[trial_number].user_attrs.get("trees")}')
    if option is None or not any(kw in option for kw in ['full','detail']):
        return models ######### just return models
    
    import numpy as np
    
    # Dictionary to hold the diagnostic results for all folds
    booster_stats = {
        'trees': [],
        'actual_max_leaves': [],
        'deepest_branch': [],
        'min_leaf_samples': [],
        'unused_features': [],
        'best_rmse' :[],
        'best_iter' :[],
    }
    
    for i, bst in enumerate(models):

        dump = bst.dump_model()
        tree_info = dump['tree_info']
        
        # --- NEW: Recursive depth counter ---
        def get_depth(node):
            if 'leaf_index' in node or 'leaf_value' in node:
                return 0
            return 1 + max(get_depth(node['left_child']), get_depth(node['right_child']))

        # 2. Extract Leaf counts and CALCULATE Real Depths
        actual_leaves = [tree['num_leaves'] for tree in tree_info]
        # Calculate depth by traversing the tree structure manually
        depths = [get_depth(tree['tree_structure']) for tree in tree_info]
        
        booster_stats['trees'].append(bst.num_trees())
        booster_stats['actual_max_leaves'].append(max(actual_leaves))
        booster_stats['deepest_branch'].append(max(depths))
        
        
        # 3. Find smallest leaf (Existing logic) DDebug
        leaf_sizes = []
        def get_leaf_counts(node):
            if 'leaf_count' in node:
                leaf_sizes.append(node['leaf_count'])
            if 'left_child' in node: get_leaf_counts(node['left_child'])
            if 'right_child' in node: get_leaf_counts(node['right_child'])
    
        for tree in tree_info:
            get_leaf_counts(tree['tree_structure'])
        
        booster_stats['min_leaf_samples'].append(min(leaf_sizes) if leaf_sizes else "N/A")
        
        # 4. Unused Features (Existing logic)
        feat_names = bst.feature_name()
        importances = bst.feature_importance(importance_type='split')
        unused_feats = [name for name, imp in zip(feat_names, importances) if imp == 0]
        booster_stats['unused_features'].append(unused_feats)
        study.trials[trial].set_user_attr("unused_features", ", ".join(unused_feats))
        study.trials[trial].set_user_attr("n_unused", len(unused_feats))
        
        # 1. Get the Best Iteration
        # In a Booster, it's .best_iteration (no underscore)
        best_iter = bst.best_iteration
        if best_iter <= 0:
            best_iter = bst.current_iteration()
        
        # 1. Official C++ Attribute Dictionary (requires parentheses)
        # Check if the method exists, then call it
        official_attrs = bst.attr() if hasattr(bst, 'attr') else {}
        if official_attrs is None: official_attrs = {} # Some versions return None if empty
        
        # 2. Manual Metadata Dictionary (your fallback from the fitting code)
        manual_attrs = getattr(bst, 'metadata', {})

        # 3. Pull values (Checking official first, then manual)
        rmse_val = "N/A" #official_attrs.get('best_rmse') or manual_attrs.get('best_rmse', "N/A")
        best_iter = official_attrs.get('best_iteration') or manual_attrs.get('best_iteration', -1)

        # 4. Conversion logic (Same as before)
        if rmse_val != "N/A": rmse_val = float(rmse_val)
        if best_iter != -1: best_iter = int(best_iter)
        
        booster_stats['best_rmse'].append(rmse_val)
        booster_stats['best_iter'].append(best_iter)      

    # --- Print Summary ---
    print(f"{'Fold':<6} | {'Trees':<6} | {'Max Leaves':<10} | {'Max Depth':<10} | {'Min Leaf Size':<15} | {'rmse':<10} | {'iter':<10} | {'Unused Feats'}")
    print("-" * 85)
    for i in range(len(models)):
        print(f"{i:<6} | "
            f"{booster_stats['trees'][i]:<6} | "
            f"{booster_stats['actual_max_leaves'][i]:<10} | "
            f"{booster_stats['deepest_branch'][i]:<10} | "
            f"{booster_stats['min_leaf_samples'][i]:<15} | "
            f"{booster_stats['best_rmse'][i]:<10} | "
            f"{booster_stats['best_iter'][i]:<10} | "
            f"{len(booster_stats['unused_features'][i])}")
    
    print(f'Successfully returned {len(models)} models from trial "{trial}".')
    return models
    

def optuna_db_to_csv(study_name='ts_0304'):
    #### working query
    import urllib.parse
    
    optunapw = os.getenv('optunapw')
    
    DB_PARAMS = {
        "host": '192.168.1.254',
        "database": 'optuna_db',
        "user": 'optuna_user',
        "password": f'{optunapw}',
        "port": '5432' # Default PostgreSQL port
    }
    
    conn_string = f"postgresql://{DB_PARAMS['user']}:{urllib.parse.quote_plus(DB_PARAMS['password'])}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"
    
    study = optuna.load_study(
        study_name=study_name, 
        storage=conn_string
    )
    
    # Get all results in one table
    df = study.trials_dataframe()
    df.columns = [col.replace('params_', '').replace('user_attrs_', '') for col in df.columns]
    nam = kg.get_nb_version()
    jk.df_to_csv(df, f'tuning {nam}.csv')

optuna_trials_to_csv = optuna_db_to_csv

def optuna_study_to_csv(study_name):
    import pandas as pd
    
    # Convert the dict to a DataFrame
    df_study_attrs = pd.DataFrame([study.user_attrs])
    
    # Export to CSV
    nam = kg.get_nb_version()
    sname = study.study_name
    df_study_attrs.to_csv(f"{sname}_study_data_{nam}.csv", index=False)


def optuna_get_all_studies():
    storage_url = optuna_conn()
    
    # 2. Get summaries
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    
    # Header for the table
    print(f"{'Study Name':<50} | {'Trials':<8} | {'Best Value':<20}")
    print("-" * 70)

    study_list = []
    
    for summary in summaries:
        # Handle studies that might not have a best trial yet
        if summary.best_trial:
            best_val = f"{summary.best_trial.value:.6f}"
            best_trial_number = summary.best_trial.number
        else:
            best_val = "N/A"
            best_trial_number = "n/a"
        study_list.append(summary.study_name)
        # summary.n_trials gives the total count (including pruned/failed)
        print(f"{summary.study_name:<80} | {summary.n_trials:<8} | {str(best_val) + ' @' + str(best_trial_number):<20}")
    return study_list
    