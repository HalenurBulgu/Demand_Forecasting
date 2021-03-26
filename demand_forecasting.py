#####################################################
# Store Item Demand Forecasting
#####################################################

# Farklı store için 3 aylık item-level sales tahmini.
# 5 yıllık bir veri setinde 10 farklı mağaza ve 50 farklı item var.
# Buna göre mağaza-item kırılımında 3 ay sonrasının tahminlerini vermemiz gerekiyor.
# hiyerarşik forecast ya da...


#####################################################
# Libraries
#####################################################

import time
import numpy as np
import pandas as pd
# pip install lightgbm
import lightgbm as lgb
import warnings
from helpers.eda import *
from helpers.data_prep import *

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


#####################################################
# Loading the data
#####################################################

train = pd.read_csv('datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('datasets/demand_forecasting/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('datasets/demand_forecasting/sample_submission.csv')
df = pd.concat([train, test], sort=False)

#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()
check_df(train)
# (913000, 4)

check_df(test)

check_df(sample_sub)

check_outlier(df, "sales")
missing_values_table(df)
#outlier ve kayıp değer bulunmamaktadır.

# Satış dağılımı nasıl?
df[["sales"]].describe().T

# Kaç store var?
df[["store"]].nunique()

# Kaç item var?
df[["item"]].nunique()

# Her store'da eşit sayıda mı eşsiz item var?
df.groupby(["store"])["item"].nunique()

# Peki her store'da eşit sayıda mı sales var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# mağaza-item kırılımında satış istatistikleri
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# FEATURE ENGINEERING
#####################################################

#####################################################
# Date Features
#####################################################

df.head()
df.shape

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    #hafta salı'dan başladığı için +1 atarız.
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    #alışverişlerde cuma-ctes-pazar önemli olduğu için 0'dan başlayarak haftanın 4. günü olan Cuma'dan itibaren çıktı 1 gelir.
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head(20)


df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})
#buradan aylık tahminler de yapılabilir.

#####################################################
# Random Noise
#####################################################

#overfitin önüne geçmek için:
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


#####################################################
# Lag/Shifted Features
#####################################################


df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

check_df(df)
df["sales"].head(10)

#sales değişkeninin bir önceki değişkenini incelemek.
df["sales"].shift(1).values[0:10]

#sales değişkenlerinin bir önceki değerlerini yanına eklemek.
pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})

#SES modelinde, moving average'da en çok kendinden bir öncekinden etkileniyordu.

df.groupby(["store", "item"])['sales'].head()

#transform ile yine aynı shift işlemini yapıyoruz.
df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))
#en son veriyi birleştirdiğimiz için çıktıda test seti olduğundan nan olarak geliyor.

def lag_features(dataframe, lags):
    dataframe = dataframe.copy()
    # gecikmelerin hepsini gez
    for lag in lags:
        #ve gezdiği gecikmeleri otomatik isimlendirerek:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


#tahmin etmek istediğimiz periyodlar 3 aylık:
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

df.head()

df[df["sales"].isnull()]

df[df["date"] == "2017-10-02"]
#tarihin 91 gün öncesini inceledik:
# pd.to_datetime("2018-01-01") - pd.DateOffset(91)

#####################################################
# Rolling Mean Features
#####################################################

# Hareketli Ortalamalar

df["sales"].head(10)
df["sales"].rolling(window=2).mean().values[0:10]
df["sales"].rolling(window=3).mean().values[0:10]
df["sales"].rolling(window=5).mean().values[0:10]

#kendisi dahil önceki değerlerin ortalaması:
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

#kendisi dahil olmamalı ki geçmişteki trendi ifade edebilecek mevcut değerden bağımsız bir feature türetilebilsin:
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

def roll_mean_features(dataframe, windows):
    dataframe = dataframe.copy()
    for window in windows:
        #otomatik isimlendirme:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)
            #window'un min_periods'dan küçük olmaması gerekiyor.
    return dataframe

df = roll_mean_features(df, [365, 546])

df.head()


#####################################################
# Exponentially Weighted Mean Features
#####################################################


pd.DataFrame({"sales": df["sales"].values[0:10],
              #karşılaştırmak için bir tane hareketli ortalama alalım:
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              #alphalar geçmiş değerlere ne kadar önem vereceğimizi belirleyen bir parametre:
              #daha yakın tarihe daha yüksek ağırlık veririz:
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm01": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})



def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    #hem ağırlıklı ortalamaya göre hem de laglara göre feature türetiyoruz:
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

#her alphas ve lags kombinasyonları ile değişken türetti:
df = ewm_features(df, alphas, lags)

check_df(df)
df.columns
#71 adet değişken.

#####################################################
# One-Hot Encoding
#####################################################

#'store', 'item', 'day_of_week', 'month' için one-hot dönüşümü yapalım.
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])


#####################################################
# Converting sales to log(1+sales)
#####################################################

#sales'e 1 eklendikten sonra logaritmik dönüşüm yapılıyor.
#çünki 0'ın logaritması alınmaz.
df['sales'] = np.log1p(df["sales"].values)

#regresyon problemlerinde bağımlı değişkene logaritmik dönüşüm yapılarak gradient descent kullanılan yöntemlerde iterasyon sayısını arttırır.

#####################################################
# Custom Cost Function
#####################################################

# MAE: mean absolute error
# MAPE: mean absolute percentage error

#büyük olan hataların dominantlığını kırmak için:
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

##
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False
##

#####################################################
# MODEL VALIDATION
#####################################################

# Light GBM: optimizasyon 2 açıdan ele alınmalı:
#Parametreler
#İterasyon sayısı

#####################################################
# Time-Based Validation Sets
#####################################################

# Kaggle test seti tahmin edilecek değerler: 2018'in ilk 3 ayı.

test["date"].min(), test["date"].max()
train["date"].min(), train["date"].max()

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]
train["date"].min(), train["date"].max()

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

df.columns

#bağımsız değişkenleri seçelim:
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

#train ve test setinin bağımlı ve bağımsız değişkenlerini tanımlayalım:
Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


#####################################################
# LightGBM Model
#####################################################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
                #her iterasyonda göz önünde bulundurulacak değişken sayısı:'feature_fraction'
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
                #n-estimators:'num_boost_round': en az 10000 yapılması tavsiye edilir.
              'num_boost_round': 1000,
                #overfitin önüne geçmek için:'early_stopping_rounds', train süresini azaltır. Validasyon hatasında artık hata düşmüyorsa dur der.
              'early_stopping_rounds': 200,
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error
# learning_rate: shrinkage_rate, eta
# num_boost_round: n_estimators, number of boosting iterations.
# nthread: num_thread, nthread, nthreads, n_jobs


#train ve validasyonu lgb formuna dönüştürelim:
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)


y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
#Logaritma alındığı için geri dönüştürerek smape'e bakıyoruz./Yüzdelik cinsten hata oranı:
smape(np.expm1(y_pred_val), np.expm1(Y_val))



##########################################
# Değişken önem düzeyleri
##########################################

def plot_lgb_importances(model, plot=False, num=10):
    """" Değişken önem düzeyini belirlerç
        split ilgili feature'ın bütün modelleme süresince kaç defa split edilmek üzere kullanıldığını gösterir.
        gain feature'ın kazandırdığı gain/gain: bölmeden önceki ve bölmedne sonraki entrpoi değişimidir."""
    from matplotlib import pyplot as plt
    import seaborn as sns
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)

plot_lgb_importances(model, plot=True, num=30)



lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()


##########################################
# Final Model
##########################################
#Bütün veriyi kullanarak modelleme yapalım:

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

#Validasyon bittiği için final modelinde early_stopping_rounds'u parametrelerden çıkarmamız gerekir.
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

# Create submission
submission_df = test.loc[:, ['id', 'sales']]
#ters logaritma alalım:
submission_df['sales'] = np.expm1(test_preds)
#ondalıklı ifadeyi int çevirelim:
submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv('submission.csv', index=False)
