import pandas as pd

# WADI
# df = pd.read_excel('WADI_attackdata_labelled.xlsx')
# df = df.drop(['Row'], axis=1)
# df['attack'] = df['attack'].map({-1: 0, 1: 1})
# df.to_csv('WADI_attackdata_labelled.csv', index=True)

# SWAT
df = pd.read_excel('SWaT_Dataset_Attack_v0.xlsx')
df['attack'] = df['attack'].map({'Normal': 0, 'Attack': 1})
df.to_csv('swat_test.csv', index=True)