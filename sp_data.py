import pandas as pd
import numpy as np

def add_eng_features(X, tag=None):

    epsilon = 1e-8
    Xf = {}
    clear_list = []

    Xf['ratio_Mo_Do3'] = X.Momentum / (X.Donchian_3 + epsilon)

    Xf['ratio_ROC_Volatility'] = (X['ROC'] / X['Volatility'] + epsilon) # MI: 0.6029 | mean: 0.0075 | std: 0.0922 | Codes: 1200
    # Xf['ratio_Momentum_Bbands_2'] = (X['Momentum'] / X['Bbands_2'] + epsilon) # MI: 0.5779 | mean: 0.0046 | std: 0.0510 | Codes: 1200
    # Xf['ratio_Momentum_Pbands_3'] = (X['Momentum'] / X['Pbands_3'] + epsilon) # MI: 0.5738 | mean: 0.0048 | std: 0.0526 | Codes: 1200
    Xf['sum_Stochastics_1_ROC'] = X['Stochastics_1'] + X['ROC'] # MI: 0.5731 | mean: 0.5596 | std: 0.4092 | Codes: 1200
    # Xf['sum_ROC_Bbands_3'] = X['ROC'] + X['Bbands_3'] # MI: 0.5713 | mean: 0.5501 | std: 0.3776 | Codes: 1200
    # Xf['ratio_Momentum_Pbands_2'] = (X['Momentum'] / X['Pbands_2'] + epsilon) # MI: 0.5683 | mean: 0.0056 | std: 0.0604 | Codes: 1200
    Xf['diff_Aroon_CCI'] = X['Aroon'] - X['CCI'] # MI: -0.5642 | mean: -9.8978 | std: 83.7474 | Codes: 1200
    # Xf['diff_CCI_CMO'] = X['CCI'] - X['CMO'] # MI: 0.5541 | mean: 8.4708 | std: 82.2138 | Codes: 1200
    # Xf['sum_CCI_Momentum'] = X['CCI'] + X['Momentum'] # MI: 0.5534 | mean: 15.8470 | std: 117.2853 | Codes: 1200
    # Xf['ratio_Momentum_Pbands_1'] = (X['Momentum'] / X['Pbands_1'] + epsilon) # MI: 0.5511 | mean: 0.0069 | std: 0.0735 | Codes: 1200
    # Xf['sum_Stochastics_1_Bbands_3'] = X['Stochastics_1'] + X['Bbands_3'] # MI: 0.5478 | mean: 1.1024 | std: 0.6982 | Codes: 1200
    Xf['sum_RSI_Momentum'] = X['RSI'] + X['Momentum'] # MI: 0.5463 | mean: 53.6500 | std: 20.1540 | Codes: 1200
    # Xf['ratio_Momentum_Bbands_1'] = (X['Momentum'] / X['Bbands_1'] + epsilon) # MI: 0.5459 | mean: 0.0073 | std: 0.0767 | Codes: 1200
    # Xf['diff_CCI_MFI'] = X['CCI'] - X['MFI'] # MI: 0.5451 | mean: -37.8840 | std: 101.3033 | Codes: 1200
    # Xf['product_MFI_ROC'] = X['MFI'] * X['ROC'] # MI: 0.5448 | mean: 0.5001 | std: 3.3318 | Codes: 1200
    # Xf['product_RSI_ROC'] = X['RSI'] * X['ROC'] # MI: 0.5429 | mean: 0.5914 | std: 3.3126 | Codes: 1200
    # Xf['diff_CCI_MACD_1'] = X['CCI'] - X['MACD_1'] # MI: 0.5424 | mean: 14.5508 | std: 109.2262 | Codes: 1200
    # Xf['diff_CCI_MACD_2'] = X['CCI'] - X['MACD_2'] # MI: 0.5420 | mean: 14.5444 | std: 110.3060 | Codes: 1200
    # Xf['diff_ADX_CCI'] = X['ADX'] - X['CCI'] # MI: -0.5392 | mean: 9.1292 | std: 108.3499 | Codes: 1200
    
    # clear_list.extend(['Pbands_2','Donchian_2','Donchian_3','EMA','SMA','Bbands_2','Pbands_1' ,'Bbands_3','ROC','CCI','Stochastics_1','Donchian_1','MACD_2','DVI_3','Pbands_3','TRIX_2','Stochastics_3','KST_2','KST_1','','']) #
    clear_list.extend(['SMA', 'VHF', 'OBV', 'KST_1', 'KST_2', 'DVI_1', 'Pbands_1','Donchian_1', 'Pbands_3', 'EMA', 'Pbands_2', 'DVI_2', 'MACD_2',      'Stochastics_3', 'Donchian_2', 'TDI_2', 'Donchian_3', 'TRIX_2','TRIX_1', 'Bbands_1', 'TDI_1', 'Bbands_2']) # let's see
    clear_list = [c for c in clear_list if c]
    
    # target conditions = ((X['y_coded'] == 1))
    # target % in population  = 37.7% - 452
    # Path 0: ((X.ROC <= 0.01860091) & (X.Stochastics_2 <= 0.94935945) & (X.ROC <= -0.01673716) & (X.Stochastics_1 <= 0.79029012)) -> Class: 0
    # Coverage: 31.4% (n=377 / 1200), Precision: 6.4% (n=24 / 377), Recall: 5.3% (n=24.0 / 452)
    # Path 1: ((X.ROC <= 0.01860091) & (X.Stochastics_2 <= 0.94935945) & (X.ROC <= -0.01673716) & (X.Stochastics_1 >  0.79029012)) -> Class: 0
    # Coverage: 2.2% (n=26 / 1200), Precision: 34.6% (n=9 / 26), Recall: 2.0% (n=9.0 / 452)
    # Path 2: ((X.ROC <= 0.01860091) & (X.Stochastics_2 <= 0.94935945) & (X.ROC >  -0.01673716) & (X.Stochastics_1 <= 0.11771522)) -> Class: 0
    # Coverage: 3.2% (n=38 / 1200), Precision: 5.3% (n=2 / 38), Recall: 0.4% (n=2.0 / 452)
    # Path 3: ((X.ROC <= 0.01860091) & (X.Stochastics_2 <= 0.94935945) & (X.ROC >  -0.01673716) & (X.Stochastics_1 >  0.11771522)) -> Class: 0
    # Coverage: 19.9% (n=239 / 1200), Precision: 26.8% (n=64 / 239), Recall: 14.2% (n=63.99999999999999 / 452)
    # Path 4: ((X.ROC <= 0.01860091) & (X.Stochastics_2 >  0.94935945) & (X.Volatility <= 0.71657017) & (X.Momentum <= 2.80834198)) -> Class: 1
    # Coverage: 4.5% (n=54 / 1200), Precision: 38.9% (n=21 / 54), Recall: 4.6% (n=21.0 / 452)
    # Path 5: ((X.ROC <= 0.01860091) & (X.Stochastics_2 >  0.94935945) & (X.Volatility <= 0.71657017) & (X.Momentum >  2.80834198)) -> Class: 1
    # Coverage: 0.8% (n=9 / 1200), Precision: 88.9% (n=8 / 9), Recall: 1.8% (n=8.0 / 452)
    # Path 6: ((X.ROC <= 0.01860091) & (X.Stochastics_2 >  0.94935945) & (X.Volatility >  0.71657017) & (X.Bbands_3 <= 0.81258169)) -> Class: 0
    # Coverage: 0.1% (n=1 / 1200), Precision: 0.0% (n=0 / 1), Recall: 0.0% (n=0.0 / 452)
    # Path 7: ((X.ROC <= 0.01860091) & (X.Stochastics_2 >  0.94935945) & (X.Volatility >  0.71657017) & (X.Bbands_3 >  0.81258169)) -> Class: 1
    # Coverage: 0.8% (n=10 / 1200), Precision: 100.0% (n=10 / 10), Recall: 2.2% (n=10.0 / 452)
    # Path 8: ((X.ROC >  0.01860091) & (X.Bbands_3 <= 0.93168634) & (X.KST_1 <= -81.80369568) & (X.Stochastics_2 <= 0.06976826)) -> Class: 0
    # Coverage: 0.2% (n=3 / 1200), Precision: 0.0% (n=0 / 3), Recall: 0.0% (n=0.0 / 452)
    # Path 9: ((X.ROC >  0.01860091) & (X.Bbands_3 <= 0.93168634) & (X.KST_1 <= -81.80369568) & (X.Stochastics_2 >  0.06976826)) -> Class: 1
    # Coverage: 8.2% (n=99 / 1200), Precision: 76.8% (n=76 / 99), Recall: 16.8% (n=76.0 / 452)
    # Path 10: ((X.ROC >  0.01860091) & (X.Bbands_3 <= 0.93168634) & (X.KST_1 >  -81.80369568) & (X.Stochastics_1 <= 0.53074244)) -> Class: 0
    # Coverage: 5.1% (n=61 / 1200), Precision: 27.9% (n=17 / 61), Recall: 3.8% (n=17.0 / 452)
    # Path 11: ((X.ROC >  0.01860091) & (X.Bbands_3 <= 0.93168634) & (X.KST_1 >  -81.80369568) & (X.Stochastics_1 >  0.53074244)) -> Class: 1
    # Coverage: 11.9% (n=143 / 1200), Precision: 60.8% (n=87 / 143), Recall: 19.2% (n=87.0 / 452)
    # Path 12: ((X.ROC >  0.01860091) & (X.Bbands_3 >  0.93168634) & (X.OBV <= 12645066240.00000000) & (X.MFI <= 51.63562393)) -> Class: 1
    # Coverage: 0.5% (n=6 / 1200), Precision: 66.7% (n=4 / 6), Recall: 0.9% (n=4.0 / 452)
    # Path 13: ((X.ROC >  0.01860091) & (X.Bbands_3 >  0.93168634) & (X.OBV <= 12645066240.00000000) & (X.MFI >  51.63562393)) -> Class: 1
    # Coverage: 11.1% (n=133 / 1200), Precision: 97.7% (n=130 / 133), Recall: 28.8% (n=130.0 / 452)
    # Path 14: ((X.ROC >  0.01860091) & (X.Bbands_3 >  0.93168634) & (X.OBV >  12645066240.00000000)) -> Class: 0
    # Coverage: 0.1% (n=1 / 1200), Precision: 0.0% (n=0 / 1), Recall: 0.0% (n=0.0 / 452)

    
    Path_9 = ((X.ROC >  0.01860091) & (X.Bbands_3 <= 0.93168634) & (X.KST_1 <= -81.80369568) & (X.Stochastics_2 >  0.06976826))
    # Coverage: 8.2% (n=99 / 1200), Precision: 76.8% (n=76 / 99), Recall: 16.8% (n=76.0 / 452)
    Path_11 = ((X.ROC >  0.01860091) & (X.Bbands_3 <= 0.93168634) & (X.KST_1 >  -81.80369568) & (X.Stochastics_1 >  0.53074244))
    # Coverage: 11.9% (n=143 / 1200), Precision: 60.8% (n=87 / 143), Recall: 19.2% (n=87.0 / 452)
    Path_13 = ((X.ROC >  0.01860091) & (X.Bbands_3 >  0.93168634) & (X.OBV <= 12645066240.00000000) & (X.MFI >  51.63562393))
    # Coverage: 11.1% (n=133 / 1200), Precision: 97.7% (n=130 / 133), Recall: 28.8% (n=130.0 / 452)

    conditions = [Path_9,Path_11,Path_13]
    values = np.array([76.8, 60.8, 97.7 ])/37.7
    # Xf['y_eq_1'] = np.select(conditions, values, default = 0)
    # Xf['y_eq_1'] = (X.ROC - 0.01860091).clip(lower=0) * (0.93168634 - X.Bbands_3).clip(lower=0)
    

    
    # target conditions = ((X['y_coded'] == 0))
    # target % in population  = 31.6% - 379
    # Path 0: ((X.Bbands_3 <= 1.02065015) & (X.RSI <= 43.54730797) & (X.ROC <= -0.03175711) & (X.MACD_2 <= -9.38750648)) -> Class: 1
    # Coverage: 0.3% (n=4 / 1200), Precision: 75.0% (n=3 / 4), Recall: 0.8% (n=3.0 / 379)
    # Path 1: ((X.Bbands_3 <= 1.02065015) & (X.RSI <= 43.54730797) & (X.ROC <= -0.03175711) & (X.MACD_2 >  -9.38750648)) -> Class: 0
    # Coverage: 13.4% (n=161 / 1200), Precision: 11.2% (n=18 / 161), Recall: 4.7% (n=18.0 / 379)
    # Path 2: ((X.Bbands_3 <= 1.02065015) & (X.RSI <= 43.54730797) & (X.ROC >  -0.03175711) & (X.ROC <= 0.01849371)) -> Class: 1
    # Coverage: 7.2% (n=86 / 1200), Precision: 38.4% (n=33 / 86), Recall: 8.7% (n=33.0 / 379)
    # Path 3: ((X.Bbands_3 <= 1.02065015) & (X.RSI <= 43.54730797) & (X.ROC >  -0.03175711) & (X.ROC >  0.01849371)) -> Class: 0
    # Coverage: 3.6% (n=43 / 1200), Precision: 7.0% (n=3 / 43), Recall: 0.8% (n=3.0 / 379)
        # Path 4: ((X.Bbands_3 <= 1.02065015) & (.RSI >  43.54730797) & (X.ROC <= 0.02732293) & (X.ROC <= -0.04835706)) -> Class: 0
    # Coverage: 6.2% (n=74 / 1200), Precision: 23.0% (n=17 / 74), Recall: 4.5% (n=17.0 / 379)
    # Path 5: ((X.Bbands_3 <= 1.02065015) & (X.RSI >  43.54730797) & (X.ROC <= 0.02732293) & (X.ROC >  -0.04835706)) -> Class: 1
    # Coverage: 40.9% (n=491 / 1200), Precision: 49.7% (n=244 / 491), Recall: 64.4% (n=244.0 / 379)
    # Path 6: ((X.Bbands_3 <= 1.02065015) & (X.RSI >  43.54730797) & (X.ROC >  0.02732293) & (X.Stochastics_1 <= 0.36925013)) -> Class: 1
    # Coverage: 2.2% (n=27 / 1200), Precision: 59.3% (n=16 / 27), Recall: 4.2% (n=16.0 / 379)
    # Path 7: ((X.Bbands_3 <= 1.02065015) & (X.RSI >  43.54730797) & (X.ROC >  0.02732293) & (X.Stochastics_1 >  0.36925013)) -> Class: 0
    # Coverage: 18.1% (n=217 / 1200), Precision: 20.3% (n=44 / 217), Recall: 11.6% (n=44.0 / 379)
    # Path 8: ((X.Bbands_3 >  1.02065015) & (X.OBV <= 12237046272.00000000)) -> Class: 0
    # Coverage: 8.0% (n=96 / 1200), Precision: 0.0% (n=0 / 96), Recall: 0.0% (n=0.0 / 379)
    # Path 9: ((X.Bbands_3 >  1.02065015) & (X.OBV >  12237046272.00000000)) -> Class: 1
    # Coverage: 0.1% (n=1 / 1200), Precision: 100.0% (n=1 / 1), Recall: 0.3% (n=1.0 / 379)

    Path_5 = ((X.Bbands_3 <= 1.02065015) & (X.RSI >  43.54730797) & (X.ROC <= 0.02732293) & (X.ROC >  -0.04835706))
    Path_6 = ((X.Bbands_3 <= 1.02065015) & (X.RSI >  43.54730797) & (X.ROC >  0.02732293) & (X.Stochastics_1 <= 0.36925013))

    conditions = [Path_5,Path_6]
    values = np.array([49.7, 59.3])/31.6
    # Xf['y_eq_0'] = np.select(conditions, values, default = 0)
    # Xf['y_eq_0'] = (1.02065015 - X.Bbands_3).clip(lower=0) * (X.RSI - 43.54730797).clip(lower=0) * (0.02732293 - X.ROC).clip(lower=0) * (X.ROC - -0.04835706).clip(lower=0)
    
    
    X = pd.concat([X, pd.DataFrame(Xf, index=X.index)], axis=1)
    X = X.loc[:, ~X.columns.duplicated(keep='last')]
    Xf = {}

    clear_list.extend(['KST_2','Volatility','TDI_2','OBV','TRIX_1','TDI_1','KST_1','Bbands_1','ADX','','','','']) #'hinge_6','hinge_2','hinge_3','hinge_4']
    X.drop(columns=clear_list, errors='ignore', inplace=True)


    return X