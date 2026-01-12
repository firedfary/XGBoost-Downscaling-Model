import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import xarray as xr

p=20

pret = pd.DataFrame(np.load('./xg6/data2/prmmod1_sstobs_pcs.npy')).T
sstt = pd.DataFrame(np.load('./xg6/data2/prmmod_sstobs1_pcs.npy')).T
slpt = pd.DataFrame(np.load('./xg6/data2/prmmod_slpmod1_pcs.npy')).T
gh5t = pd.DataFrame(np.load('./xg6/data2/prmmod_gh5mod1_pcs.npy')).T

tart = pd.DataFrame(np.load('./xg6/data2/prmobs1_sstobs_pcs.npy')).T
c = pd.DataFrame(np.concatenate((sstt, slpt, gh5t, pret), axis=1))
# c = pret

kf = KFold(n_splits=23)
a1 = 1
all_preds = []
for train_index, test_index in kf.split(c):
    train_x, test_x = c.iloc[train_index], c.iloc[test_index]
    train_y, test_y = tart.iloc[train_index], tart.iloc[test_index]

    dtrain=xgb.DMatrix(train_x,label=train_y)
    dtest=xgb.DMatrix(test_x)
    watchlist = [(dtrain,'train')]
    # dal = xgb.DMatrix(x_value)

    # booster:
    params={'booster':'gblinear',
            'objective': 'reg:squarederror',
            'max_depth':1,
            'min_child_weight':2,
            'gamma':0,
            'subsample':1,
            'colsample_bytree':1,
            'alpha':0,
            'lambda':0,
            'eval_metric':'mae',
            'seed':3,
            'nthread':8,
            'learning_rate' : 0.2}


    bst=xgb.train(params, dtrain,num_boost_round=25,evals=watchlist)
    
    # bst.save_model('./xg8infer/data2/'+str(a1)+'.json')
    a1 += 1

    # 评价 训练集中y的预测值和真实值
    y_pred = bst.predict(dtest)
    all_preds.append(y_pred)

all_preds_concatenated = np.concatenate(all_preds,axis=0)
np.save('./xg6/data_pred/pre.npy', all_preds_concatenated)
print('ok')