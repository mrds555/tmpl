
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import polars as pl
import sys

lib_path1 = Path(os.getenv('WS')) / 'sync_share' / 'python' / 'lib'
lib_path2 = Path(os.getenv('WS')) / 'sync_share' / 'kaggle' / 'lib'
lib_path3 = Path(os.getenv('WS')) / 'sync_share' / 'kg' / 'lib'
sys.path.append(str(lib_path1))
sys.path.append(str(lib_path2))
sys.path.append(str(lib_path3))

import libdf_jk as dfjk, libplot_jk as pltjk, kg_jk as kg # ts_data as tsd


def help():
    output = '''
import_libs()
showlib(lib_name)
df_hygiene()
clock()
col_summary(df, column)
get_timestamp() #yyyymmdd_hhmmss string
cat_code(df, col, val=None)

#jupyter notebook:
%store # see stored variables
%store variable_name #save from notebook A
%store -r variable_name #save from notebook B
%store -d variable_name #permanently delete the variable from the %store database

del your_variable_name #clear from active kernel
%reset # clear all user-defined variables
pickle.dump(obj, open('file.pkl', 'wb')) to save almost any Python object.


    '''
    print(output)

def clist(input_list):

    input_list[:] = [i for i in input_list if i]
    return input_list
    

def startup():
    import os
    # Replace with the path found by 'ipython locate profile'
    startup_file = r'd:/jk/myenv312/profile_default/startup/00_startup.py'
    # %run -i {startup_file}



def re_str_to_list(raw_string, incl='', excl=''):
    import re

    # raw_string = "Feature_A!!\n  feature-b,   feature.c\t[92]"
    
    # \W+ matches any non-word character (anything not a-z, 0-9, or _)
    # We replace them with a space, then split to remove empty strings
    clean_list = re.sub(r'\W+', ' ', raw_string).split()

    # alt 1
    clean_string = re.sub(r'[^a-zA-Z0-9_]', ' ', raw_string)
    keywords = clean_string.split()

    # alt 2
    keywords = re.findall(r'[a-zA-Z][a-zA-Z0-9_]{2,}', raw_string)
    
    return clean_list

def import_libs():
    import importlib
    import sys
    
    WS = os.getenv('WS')
    
    path = Path(WS) / 'sync_share/kg/lib'
    if path not in sys.path:
        sys.path.insert(0, str(path))
    
    path = Path(WS) / 'sync_share/python/lib'
    if path not in sys.path:
        sys.path.insert(0, str(path))
        
    import lib_jk as jk
    importlib.reload(jk)
    
    import kg_jk as kg
    importlib.reload(kg)
    
    import libdf_jk as dfjk
    importlib.reload(dfjk)
    
    import libplot_jk as pltjk
    importlib.reload(pltjk)

    import lib_jk_b as jkb
    importlib.reload(jkb)

libs = import_libs

def reload_lib(lib_names):
    import importlib
    
    if not isinstance(lib_names, (list, tuple, pd.Series, np.ndarray)):
        lib_names = [lib_names]
    # Returns a list of (name, function_object) tuples
    for i in lib_names:
        importlib.reload(i)


def showlib(lib_names):
    import inspect

    if not isinstance(lib_names, (list, tuple, pd.Series, np.ndarray)):
        lib_names = [lib_names]
    # Returns a list of (name, function_object) tuples
    for i in lib_names:
        functions = inspect.getmembers(i, inspect.isfunction)
    
        # To get just the names:
        func_names = [f[0] for f in functions]
        print(func_names)

    print("To get parameters, use function_params(func),")
    print("or use funcparams(func).")

def funcparams(my_function):    
    import inspect
    sig = inspect.signature(my_function)

    # Get just the parameter names as a list
    param_names = list(sig.parameters.keys())
    print(param_names)

def function_params(func):
    import inspect
    
    sig = inspect.signature(func)

    # Create a dictionary of name: default_value
    # Note: We filter out parameters that have no default value
    defaults = {
        name: param.default 
        for name, param in sig.parameters.items() 
        # if param.default is not inspect.Parameter.empty
    }
    
    print(defaults)

def df_hygiene(df):
    hygiene = pd.DataFrame({
    'dtype': df.dtypes,
    'non_null': df.count(),
    'nan_count': df.isnull().sum(),
    'nan_pct': (df.isnull().sum() / len(df)) * 100 
    })
    print(hygiene)

    
def df_to_csv(df, file_path, index=False):
    write_header = not os.path.exists(file_path)
    df.to_csv(file_path, mode='a', index=index, header=write_header)

def df_to_pq(df, file_path):
    import pyarrow
    df.to_parquet(file_path, index=False, engine='pyarrow')

def csv_to_df(csv):
    return pd.read_csv(csv)

def sample_df():
    data = {
    'resource_id': ['A', 'A', 'A', 'B', 'B'],
    'start_time': ['2026-05-01 10:00', '2026-05-01 12:00', '2026-05-01 11:00', '2026-05-02 09:00', '2026-05-02 10:00'],
    'end_time': ['2026-05-01 11:30', '2026-05-01 13:00', '2026-05-01 12:30', '2026-05-02 10:00', '2026-05-02 11:00']
    }
    df = pd.DataFrame(data)
    return df

def pq_to_df(pq_path, df_pd=None):
    import polars as pl

    # 1. This is a Polars DF
    df_pl = pl.read_parquet(pq_path)
    # 2. Convert to Pandas
    if df_pd is None:
        df_pd = pd.DataFrame()
        df_pd = df_pl.to_pandas()
    elif not isinstance(df_pd, pd.DataFrame):
        return df_pl.to_pandas()
    else:
        df_pd = df_pl.to_pandas()

    return df_pd

def pl_to_df(pl_df: pl.DataFrame) -> pd.DataFrame:
    """Converts a Polars DataFrame to a Pandas DataFrame."""
    return pl_df.to_pandas()

def pl_to_csv(pl_df, file_path):
    import polars as pl
        
    # Check for Eager DataFrame
    if isinstance(pl_df, pl.DataFrame):
        pl_df.write_csv(file_path)
    # Check for LazyFrame
    elif isinstance(pl_df, pl.LazyFrame):
        pl_df.collect().write_csv(file_path)


def clock(label="Process"):
    """Call once to start, call again to print duration."""
    if not hasattr(clock, 'start_time'):
        clock.start_time = time.perf_counter()
        print(f"--- {label} started ---")
    else:
        duration = time.perf_counter() - clock.start_time
        print(f"--- {label} finished in {duration:.4f} seconds ---")
        del clock.start_time # Reset for next use

def get_timestamp():
    from datetime import datetime

    """Generates a timestamp in yyyymmdd_hhmmss 24-hour format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def full_display():
    # pd.set_option('display.float_format', lambda x: f'{x / 1e9:.2f}B')
    # pd.reset_option('display.float_format')
    # pd.reset_option("all")
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', 100)
    # pd.set_option('display.width', 300)

    # 1. Float format (Optional: only if you want the 'B' notation right now)
    # pd.options.display.float_format = lambda x: f'{x / 1e9:.2f}B'
    
    # 2. The "Show Everything" settings
    pd.options.display.max_columns = None
    pd.options.display.max_rows = 50
    pd.options.display.min_rows = 20
    pd.options.display.width = 300
    pd.options.display.max_colwidth = None  # Prevents long strings from being cut off (...)

    # pro way
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 300):
    #     print(df_to_check)
# [job cat dsc / num pay, cat desc / cat_id, unit_eng, pay_item_desc / pay_item_id, primary loc

def sys_path_remove_last(path_to_remove):
    import sys
    
    for i in range(len(sys.path) - 1, -1, -1):
        if sys.path[i] == path_to_remove:
            del sys.path[i]
            break  # Remove only the last one and stop

############################### visualizing ##############################



def plot(X, y):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Combine into a temp df to keep X and y aligned
    target_name = y.name if hasattr(y, 'name') else 'target'
    df_plot = pd.concat([X, y], axis=1)

    for col in X.columns:
        plt.figure(figsize=(10, 5))
        
        # 1. Handle Categorical / Object Data
        if isinstance(df_plot[col].dtype, pd.CategoricalDtype) or df_plot[col].dtype == 'object':
            sns.boxenplot(data=df_plot, x=col, y=target_name)
            plt.xticks(rotation=45) # Prevents label overlap
            plt.title(f"Categorical Distribution: {col} vs {target_name}")

        # 2. Handle Datetime Data
        elif pd.api.types.is_datetime64_any_dtype(df_plot[col]):
            sns.lineplot(data=df_plot, x=col, y=target_name)
            plt.title(f"Time Series Trend: {col} vs {target_name}")

        # 3. Handle Numerical Data
        else:
            sns.scatterplot(data=df_plot, x=col, y=target_name, alpha=0.3)
            plt.title(f"Scatter Plot: {col} vs {target_name}")

        plt.tight_layout()
        plt.show()

# Usage:
# plot_all_relationships(X, y)

def colplot(col1, col2):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    
    # 1. Handle Categorical / Object Data
    if isinstance(col1.dtype, pd.CategoricalDtype) or col1.dtype == 'object':
        sns.boxenplot(x=col1, y=col2)
        plt.xticks(rotation=45) # Prevents label overlap
        plt.title(f"Categorical Distribution: {col1.name} vs {col2.name}")

    # 2. Handle Datetime Data
    elif pd.api.types.is_datetime64_any_dtype(col1):
        sns.lineplot(x=col1, y=col2)
        plt.title(f"Time Series Trend: {col1.name} vs {col2.name}")

    # 3. Handle Numerical Data
    else:
        sns.scatterplot(x=col1, y=col2, alpha=0.3)
        plt.title(f"Scatter Plot: {col1.name} vs {col2.name}")

    plt.tight_layout()
    plt.show()




def plot_legacy(X, y, xlabel=None, ylabel=None, title=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 5))

    for c in X.columns:
        sns.scatterplot(x=X[c], y=y, alpha=0.3)
        # # plt.scatter(X[c], y, alpha=0.3)
        # xlabel = c
        
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.title(title)
        
        # plt.tight_layout()
        # plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming X is your feature DataFrame and y is your target Series
def plot_relationships(X, y):
    # Ensure y is a Series for easy plotting
    target_name = y.name if hasattr(y, 'name') else 'target'
    temp_df = pd.concat([X, y], axis=1)
    
    for col in X.columns:
        plt.figure(figsize=(8, 5))
        
        if X[col].dtype.name in ['category', 'object', 'bool']:
            # Categorical vs Numerical/Categorical
            sns.boxplot(data=temp_df, x=col, y=target_name)
            plt.title(f'Box Plot: {col} vs {target_name}')
        else:
            # Numerical vs Numerical
            sns.scatterplot(data=temp_df, x=col, y=target_name, alpha=0.5)
            plt.title(f'Scatter Plot: {col} vs {target_name}')
            
        plt.tight_layout()
        plt.show()

# Usage:
# plot_relationships(X, y)

def shap_summary(X, model):
    import shap
    
    shap.initjs()
    
    X_shap = X.copy()
    
    for col in X_shap.select_dtypes(include=["category"]).columns:
        X_shap[col] = (
            X_shap[col]
            .cat.add_categories("Missing")
            .fillna("Missing")
        )
    # If using XGBoost / LightGBM / sklearn tree model
    if isinstance(model, list) and len(model) > 1:
        explainer = shap.TreeExplainer(model[0])
    else:
        explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_shap)
    shap.summary_plot(shap_values, X_shap)

def pdp(X, y, features, model):

    import matplotlib.pyplot as plt
    from xgboost import XGBRegressor
    from sklearn.inspection import PartialDependenceDisplay
    
    fig, axes = plt.subplots(
        nrows=len(features), 
        figsize=(8, 3 * len(features))
    )

    if isinstance(model, list) and len(model) > 1:
        m = (model[0])
    else:
        m = model
    
    display = PartialDependenceDisplay.from_estimator(
        m,
        X,
        features,
        kind="average",
        ax=axes
    )
    
    # Individualize each subplot scale
    for axis in display.axes_.ravel():
        for line in axis.lines:
            y = line.get_ydata()
            axis.set_ylim(
                y.min() - 0.05 * abs(y.min()),
                y.max() + 0.05 * abs(y.max())
            )
    
    plt.tight_layout()
    plt.show()

def log_filter(message=None):
    
    # This targets the specific LightGBM logger
    logger = logging.getLogger('lightgbm')
    logger.setLevel(logging.INFO)
    
    class GainFilter(logging.Filter):
        def filter(self, record):
            # Return False if the message contains our "annoying" string
            return "message" not in record.getMessage()
    
    logger.addFilter(GainFilter())

def find_nums_in_string (text):
    import re
    
    # text = """... your text with 16 numbers here ..."""
    
    # This regex finds integers and floats (e.g., 10, -5.3, 1.39e13)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    
    # Convert to floats and ensure you only take the first 16
    numbers_list = [float(n) for n in numbers[:16]]
    
    return numbers_list

def col_summary(df, column):
    stats = df[column].describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99,])
    missing_pct = df[column].isna().mean() * 100
    
    # Combine them into one Series
    summary = pd.concat([pd.Series({'missing_%': missing_pct}), stats])
    # Usage:
    # print(summary)
    
    return summary

coldesc = col_summary

def filter_list_names(input_list, excl=None, incl=None):
    if excl is not None and len(excl) > 0:
        if isinstance(excl, list):
            filter_excl = excl
        else:
            filter_excl = [excl]
        filtered_list = list(set(input_list) - set(filter_excl))
        # filtered_list = [c for c in input_list if not any(x in c for x in filter_excl)]
    else:
        filtered_list = input_list

    if incl is not None and len(incl) > 0:
        if isinstance(incl, list):
            filter_incl = incl
        else:
            filter_incl = [incl]
        filtered_list = list(set(filtered_list) | set([c for c in input_list if any(x in c for x in filter_incl)]))
    
    
    return filtered_list


def filter_list(input_list, excl=None, incl=None):
    if excl is not None and len(excl) > 0:
        if isinstance(excl, list):
            filter_excl = excl
        else:
            filter_excl = [excl]
        filtered_list = [c for c in input_list if not any(x in c for x in filter_excl)]
    else:
        filtered_list = input_list

    if incl is not None and len(incl) > 0:
        if isinstance(incl, list):
            filter_incl = incl
        else:
            filter_incl = [incl]
        filtered_list = list(set(filtered_list) | set([c for c in input_list if any(x in c for x in filter_incl)]))
    
    
    return filtered_list


def cat_code(df, col, val=None):
    """
    df: DataFrame
    col: String name of the categorical column
    val: Int (cat_code) or String (category name)
    """
    # Ensure the column is categorical
    if not isinstance(df[col].dtype, pd.CategoricalDtype):
        return "Column is not categorical."

    # Get the ordered list of names (index of this list = cat_code)
    mapping = df[col].cat.categories

    # Case 1: No input - return the full mapping as a dictionary
    if val is None:
        return dict(enumerate(mapping))

    # Case 2: Input is an integer (Code -> Name)
    if isinstance(val, int):
        try:
            return mapping[val]
        except IndexError:
            return f"Code {val} out of range."

    # Case 3: Input is a string (Name -> Code)
    if isinstance(val, str):
        try:
            # .get_loc returns the index (the integer code)
            return mapping.get_loc(val)
        except KeyError:
            return f"Name '{val}' not found."

import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # If using LightGBM/XGBoost, ensure their internal seed is also set