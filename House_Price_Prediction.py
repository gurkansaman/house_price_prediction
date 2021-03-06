######################################
# House Price Prediction
######################################

# Importing Libraries:

# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter("ignore", category=ConvergenceWarning)

import warnings
from catboost import CatBoostRegressor
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import numpy as np
from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

######################################
# Exploratory Data Analysis
######################################

train = pd.read_csv("datasets/house_prices/train.csv")
test = pd.read_csv("datasets/house_prices/test.csv")

# Kontrol edelim:
train.shape
train.info()
test.shape
test.info()

##############
# Birle??tirme
###############

# Train ve test i??in concat da kullan??labilir:
#df = pd.concat([train, test], axis=0)
#df.shape

# Train ile test birle??tirelim:
df = train.append(test).reset_index(drop=True)
df.head()
#Target de??i??keninin da????l??m??na bakt??k gerekir ise kesilebilir:

plt.hist(train["SalePrice"], bins=20)
plt.show()

# df kontrol edelim
check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# categoric ama cardinal olan veriyi neymi?? bakal??m:
cat_but_car

# 25 farkl?? de??er var:
df["Neighborhood"].nunique()

# New Feature high_corr g??re ??retildi ama iyi sonu?? al??nmad??:

df["GrLivArea*GarageArea"] = df["GrLivArea"] * df["GarageArea"]
df["GrLivArea*LotArea"] = df["GrLivArea"] * df["LotArea"]
df.replace([np.inf, -np.inf], 0, inplace=True)

##################
# Kategorik De??i??ken Analizi
##################

# cat

for col in cat_cols:
    cat_summary(df, col)


# cat but car
for col in cat_but_car:
    cat_summary(df, col)



##################
# Say??sal De??i??ken Analizi
##################


df[num_cols].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.99]).T


for col in num_cols:
     num_summary(df, col, plot=False)



##################
# Target Analizi
##################
# Corelasyon analizi:

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T


# Sat???? fiyat?? (target) g??re korelasyon hesaplanm????t??r:

def find_correlation(dataframe, numeric_cols, target , corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "target":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

low_correlations, high_correlations = find_correlation(df, num_cols,"SalePrice", corr_limit=0.60)


# Do??rusal modellerde y??ksek korele olanlar?? eleyebiliriz. Feature engineering yapabiliriz asl??nda

high_correlations

low_correlations


# Korelasyon Matris incelenmesi:
corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

######################################
# Data Preprocessing & Feature Engineering
######################################


##################
# Rare Encoding
##################

rare_analyser(df, "SalePrice", cat_cols)


df = rare_encoder(df, 0.01)

rare_analyser(df, "SalePrice", cat_cols)

# Bir column da 2 kategori olan ve 1 categori y??zdesi ??ok k??????k
# Yani entropy d??????k column = useless i??aretlenir:

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]

# Kontrol edelim:
useless_cols

# Cat_Cols g??ncelleyelim:
cat_cols = [col for col in cat_cols if col not in useless_cols]

# useless column ????karal??m:
for col in useless_cols:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice", cat_cols)

##################
# Label Encoding & One-Hot Encoding
##################

# cat_but_car ekledik:
cat_cols = cat_cols + cat_but_car

# cat_but_car bu kategoride neler var:
# ['Neighborhood']

# one_hot_encoder burada binary girmeden drop ile kullan??labilir:
df = one_hot_encoder(df, cat_cols, drop_first=True)

#Kontrol edelim:
check_df(df)

# Yenileyelim:
cat_cols, num_cols, cat_but_car = grab_col_names(df)

useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

# Analiz edelim:
for col in useless_cols_new:
    cat_summary(df, col)

rare_analyser(df, "SalePrice", useless_cols_new)

##################
# Outliers
##################

outlier=[]
for col in num_cols:
    if check_outlier(df, col,q1=0.25,q3=0.75) ==True:
        outlier.append(col)
    print(col, check_outlier(df, col,q1=0.25,q3=0.75))

outlier

for col in outlier:
    replace_with_thresholds(df,col)

##################
# Missing Values
##################

missing_values_table(df)

# null olan de??erler bulunur:
# df[col].isnull().sum() > 0

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]

# Knn ile doldural??m:
df_knn = df.select_dtypes(include=["float64","int64"])
imputer = KNNImputer(n_neighbors=15)
df_knn = imputer.fit_transform(df_knn)
df_knn = pd.DataFrame(df_knn,columns=num_cols)

for col in na_cols:
    df[col] = df_knn[col]

#Minmax Scale:
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Num_cols i??inden ????karal??m:
num_cols.remove("SalePrice") #Target de??i??keni ????kard??k
num_cols.remove("Id") #Id de??i??keni ????kard??k

for col in num_cols:
    transformer = MinMaxScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])


# Helper i??inde Fonksiyonla????rd??m:
# missing_fillna(df, "SalePrice",method="median")
#df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

######################################
# Modeling
######################################

# Test ve train birle??tirmi??tik birlikte preprocessing etmi??tik
# Sonra da salesprice null olanlar?? test olarak ay??rd??k
# Saleprice null olmayanlar?? da train olarak ay??rd??k

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)


# Hem normal hemde log(x+1) g??re d??n??????m yap??p ??al????t??r??lm????t??r:
# Bu ??ekilde log(0) patlamas?? engellenmi?? olur

# y = train_df["SalePrice"]

y = np.log1p(train_df['SalePrice'])

# X de zaten SalePrice yok!
X = train_df.drop(["Id", "SalePrice"], axis=1)

##################
# Base Models
##################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          #("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# Base sonu??lar:
#RMSE: 0.1276 (Ridge)
#RMSE: 0.3792 (Lasso)
#RMSE: 0.3792 (ElasticNet)
#RMSE: 0.1595 (KNN)
#RMSE: 0.1996 (CART)
#RMSE: 0.1376 (RF)
#RMSE: 0.1213 (SVR)
#RMSE: 0.1262 (GBM)
#RMSE: 0.1273 (LightGBM)
#RMSE: 0.1163 (CatBoost)

##############################
# Hyperparameter Optimization
##############################

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}


lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

rmse

#The result:
# 0.12075620160634011

#######################################
# Feature Selection
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


# T??m hespini grafikte g??sterir:
#plot_importance(final_model, X)

# Buras?? finalize edilmi?? hali:
plot_importance(final_model, X, 30)

# Kontrol edelim:
X.shape

# Top 30 g??re yeni feature olu??turalabilir:
top_30 = pd.DataFrame()
feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})
top_30["Feature"] = feature_imp.sort_values(by="Value", ascending=False,
                                            ignore_index=True)[:30]["Feature"]

# Kontrol edelim:
top_30["Feature"]

###################################
# Bu k??s??mda zero olarak isimlendirdi??imiz k??s??m ??nem derecesi d??????k olanlar?? ????karaca????z
# ??nemli olabilecek feature se??ilmesi:

feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})

# ??nem de??erlerine g??re hist ??izilmesi:
num_summary(feature_imp, "Value", True)

# 0 dan b??y??k olan 162 feature vard??r: ??nemli
feature_imp[feature_imp["Value"] > 0].shape

# Importance de??eri 1 den k??????k ??nemsiz de??i??kenler:
# Bunlar?? bir de??i??kene atanmas??
feature_imp[feature_imp["Value"] < 10].shape
zero_imp_cols = feature_imp[feature_imp["Value"] < 10]["Feature"].values

# X i??inde bu 1 den k??????klerin temizlenmesi:
selected_cols = [col for col in X.columns if col not in zero_imp_cols]


#######################################
# Hyperparameter Optimization with Selected Features
#######################################

lgbm_model = LGBMRegressor(random_state=46)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X[selected_cols], y)


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X[selected_cols], y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X[selected_cols], y, cv=5, scoring="neg_mean_squared_error")))

rmse

# The result:
# 0.12045387361709936


#######################################
# Sonu??lar??n Y??klenmesi
#######################################

# DataFrame olu??turulmas??:
submission_df = pd.DataFrame()

# Test datas??n??n id eklenmesi
submission_df['Id'] = test_df["Id"]

# test df i??in belirlenen cols eklenmesi
y_pred_sub = final_model.predict(test_df[selected_cols])

# log1 ters d??n??????m yap??lmas??
y_pred_sub = np.expm1(y_pred_sub)

# ters d??n??????m eklenmesi:
submission_df['SalePrice'] = y_pred_sub

# csv olarak kaydedilmesi:
submission_df.to_csv('submission.csv', index=False)


























