import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import xarray as xr

p=20

pret = pd.DataFrame(np.load('./data2/prmmodpcs.npy'))#33,20
sstt = pd.DataFrame(np.load('./data2/sstmodpcs.npy'))#33,20
slpt = pd.DataFrame(np.load('./data2/slpmodpcs.npy'))#33,20
gh5t = pd.DataFrame(np.load('./data2/gh5modpcs.npy'))#33,20

tart = pd.DataFrame(np.load('./data2/prmobspcs.npy'))#31,20
c = pd.DataFrame(np.concatenate((sstt, slpt, gh5t, pret), axis=1))
# c = pret


n_total = len(c)
n_target = n_total - len(tart)
if n_target < 0:
    raise ValueError('tart is longer than c; cannot split as requested')

target_idx = list(range(n_total - n_target, n_total)) if n_target > 0 else []
test_idx = list(range(len(tart) - 3, len(tart)))
train_idx = list(range(0, len(tart) - 3))

# Leave-one-out CV on the training set
kf = KFold(n_splits=len(train_idx))
train_preds = np.zeros((len(train_idx), tart.shape[1]))

params={'booster':'gbtree',
        'objective': 'reg:squarederror',
        'max_depth':2,
        'min_child_weight':12.6,
        'gamma':100,
        'subsample':1,
        'colsample_bytree':0,
        'alpha':1,
        'lambda':1,
        'eval_metric':'rmse',
        'nthread':8,
        'learning_rate' : 0.2}

for train_sub_idx, val_sub_idx in kf.split(train_idx):
    train_sub = [train_idx[i] for i in train_sub_idx]
    val_sub = [train_idx[i] for i in val_sub_idx]

    train_x, val_x = c.iloc[train_sub], c.iloc[val_sub]
    train_y = tart.iloc[train_sub]

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x)
    watchlist = [(dtrain, 'train')]

    bst = xgb.train(params, dtrain, num_boost_round=60, evals=watchlist)

    y_pred = bst.predict(dval)
    # Leave-one-out: val_sub_idx length is 1
    train_preds[val_sub_idx, :] = y_pred

# Retrain on full train set; predict test + target
final_train_x = c.iloc[train_idx]
final_train_y = tart.iloc[train_idx]
final_dtrain = xgb.DMatrix(final_train_x, label=final_train_y)
final_watchlist = [(final_dtrain, 'train')]
final_bst = xgb.train(params, final_dtrain, num_boost_round=60, evals=final_watchlist)

test_target_idx = test_idx + target_idx
test_target_x = c.iloc[test_target_idx]
test_target_pred = final_bst.predict(xgb.DMatrix(test_target_x))

# Restore predictions to original order
all_preds_concatenated = np.zeros((n_total, tart.shape[1]))
all_preds_concatenated[train_idx, :] = train_preds
all_preds_concatenated[test_target_idx, :] = test_target_pred
# np.save('./xg6/data_pred/pre.npy', all_preds_concatenated)
print('ok')



k = np.asarray(all_preds_concatenated)
r = np.load('./data2/prmobseof.npy')*(-1)

# k: (time, mode) = (23, 20), r: (mode, space) = (20, 2324)
# Reconstruct x(time, space) = sum_mode EOF(mode, space) * PC(time, mode)
x = np.zeros((k.shape[0], r.shape[1]))
for i in range(r.shape[0]):  # loop over modes
    a = r[i, :]              # EOF for mode i, shape (space,)
    b = k[:, i]              # PC for mode i across time, shape (time,)
    e = np.zeros((k.shape[0], r.shape[1]))
    for j in range(len(b)):
        bj = float(b[j])
        e[j, :] = a * bj
    x = x + e

prm_coord = xr.open_dataarray('./prm_coord.nc')
# Use spatial coords from observations, but override time to match prediction length
time_coord = range(1994, 1994 + k.shape[0])
space_coord = prm_coord.coords['Stn_No']
x1 = xr.DataArray(
    x,
    dims=('time', 'Stn_No'),
    coords={'time': time_coord, 'Stn_No': space_coord},
)

df = pd.read_csv('./data2/summer_pr.csv')
mean_precipitation = df.groupby('Stn_No')['Precip'].mean()
df['obs_anomaly'] = df.groupby('Stn_No')['Precip'].transform(lambda x: x - x.mean())
df['obs_percent'] = df.apply(lambda row: (row.obs_anomaly / mean_precipitation[row['Stn_No']]), axis=1)

x1_df = x1.to_dataframe(name='prep')
x1_df.reset_index(inplace=True)

df = pd.merge(df, x1_df, on=['Stn_No', 'time'], how='left')
df['X'] = df['obs_percent']
df['Y'] = df['prep']

df['XY'] = df['X'] * df['Y']
sum_XY = df.groupby('time')['XY'].sum()

df['X_square'] = df['X'] ** 2
df['Y_square'] = df['Y'] ** 2
sum_X_square = df.groupby('time')['X_square'].sum()
sum_Y_square = df.groupby('time')['Y_square'].sum()

# multiply and take square root
sqrt_sum = (sum_X_square * sum_Y_square) ** 0.5
print(df)


result = sum_XY / sqrt_sum
zero_down = len(result[result < 0])
print(result)
print(zero_down)
print(result.mean())
# result.to_csv('./观测预测acc2.csv')
# df.to_csv('./预测.csv', index=False)


# Save yearly prediction files to ./data2/pre (e.g., 1994.csv)
import os

out_dir = './data2/pre'
os.makedirs(out_dir, exist_ok=True)

pred_df = x1.to_dataframe(name='prep').reset_index()
for year, g in pred_df.groupby('time'):
    out_path = os.path.join(out_dir, f'{int(year)}.csv')
    g.to_csv(out_path, index=False)
