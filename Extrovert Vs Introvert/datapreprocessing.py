from random import randint
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class Dataprocessing:
    """
    Class defined to handle data preprocessing
    @handle_missing: to impute missing values
    @handle_categorical: Encode categorical columns
    """
    def __init__(self, dataframe, target_col):
        self.df = dataframe
        self.target_col = target_col
        self.imputed_col_cat = []
        self.imputed_col_num = []
        self.all_cols = list(self.df.columns)
        self.le_encode = LabelEncoder()
        self.oe_encode = OneHotEncoder(handle_unknown='ignore')
        self.low_cardinality = []
        self.high_cardinality = []
        self.randomint = randint(0, 50)

    def handle_missing(self):
        """
        Imputes categorical columns with most_frequent
        And numerical columns with median strategy
        """
        imputed_col_cat = []
        imputed_col_num = []
        for col in self.all_cols:
            if self.df[col].isna().sum() > 0:
                if self.df[col].dtype == 'object':
                    simimpute = SimpleImputer(strategy="most_frequent")
                    self.df[col] = simimpute.fit_transform(self.df[[col]]).ravel()
                    imputed_col_cat.append(col)
                else:
                    simimpute = SimpleImputer(strategy="median")
                    self.df[col] = simimpute.fit_transform(self.df[[col]]).ravel()
                    imputed_col_num.append(col)
            else:
                pass
        return self.df

    def handle_categorical(self):
        """
        Encode columns
        Low cardinality with LabelEncoder
        High cardinality with OneHotEncoder
        """
        self.df = self.handle_missing()
        for col in self.all_cols:
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() <= 3:
                    self.df[col] = self.le_encode.fit_transform(self.df[col].astype(str))
                    self.low_cardinality.append(col)
                else:
                    # OneHotEncoder requires 2D input and returns array, so handle properly
                    encoded = self.oe_encode.fit_transform(self.df[[col]].astype(str))  # 2D input
                    encoded_df = pd.DataFrame(
                        encoded.toarray(),  # convert sparse matrix to dense
                        columns=[f"{col}_{cat}" for cat in self.oe_encode.categories_[0]],
                        index=self.df.index
                    )
                    # Drop original column and add new one-hot columns
                    self.df.drop(columns=[col], inplace=True)
                    self.df = pd.concat([self.df, encoded_df], axis=1)
                    self.high_cardinality.append(col)
            elif self.df[col].dtype =='int64' or 'float64':
                pass
            else:
                print(self.df[col])
                raise ValueError(f"Column '{col}' has unsupported dtype: {self.df[col].dtype}")
        return self.df

    def split(self):
        """
        Split dataset into Train and Test
        """
        self.df = self.handle_categorical()
        x_final = self.df.drop(self.target_col, axis = 1)
        target_y = self.df[self.target_col]
        scaler_sc = StandardScaler()
        x_scaled = scaler_sc.fit_transform(x_final)
        x_train,x_test,y_train,y_test = train_test_split(x_scaled,target_y,test_size=0.2,
                                                         random_state = self.randomint)
        return x_train,x_test,y_train,y_test
