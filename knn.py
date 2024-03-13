import pandas as pd

# Загружаем данные
train_df = pd.read_csv("./titanic/train.csv")
print(train_df.head(), 20)

# Избавляемся от двух столбцов без нужной информации
train_df = train_df.drop(columns=["PassengerId"])
from sklearn.neighbors import KNeighborsClassifier

predictors = ["Age", "Fare"]
outcome = "Survived"

new_record = train_df.loc[0:0, predictors]
X = train_df.loc[1:, predictors]
y = train_df.loc[1:, outcome]

kNN = KNeighborsClassifier(n_neighbors=20)
kNN.fit(X, y)
kNN.predict(new_record)
print(kNN.predict_proba(new_record))

# [результат/вывод]: [[0.7 0.3]]
