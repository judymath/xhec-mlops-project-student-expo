import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def preprocessing(df):

    df.columns = [col.replace(" ", "_").lower() for col in df.columns]
    encoder = OneHotEncoder(handle_unknown='ignore')
    df_cat = encoder.fit_transform(df.select_dtypes(include='object')).toarray()
    df_num = df.select_dtypes(include='number')
    data = pd.concat([df_num, pd.DataFrame(df_cat, columns=encoder.categories_[0].tolist())], axis=1)

    X = data.drop('rings', axis=1)
    y = data['rings']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test