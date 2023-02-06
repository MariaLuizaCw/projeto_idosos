import pandas as pd
from openpyxl import load_workbook
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import export_text
import numpy as np
import os
import io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import math
import itertools
from functools import reduce
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score


import warnings
warnings.filterwarnings('ignore')

from scipy import stats

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR
from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

# pd.set_option("display.max_columns", None)
# pd.reset_option('max_rows')


class idososClass:
    def __init__(self, df: pd.DataFrame, y_col: str, x_cols: list, cat_cols: list, num_cols: list, output_path: str):
        self.df = df
        self.y_col = y_col
        self.x_cols = x_cols
        self.cat_cols = cat_cols
        self.models = dict()
        self.output_path = output_path
        self.logs = dict()
        self.num_cols = num_cols
        self.nan_lines = df.loc[:, x_cols].isna().apply(lambda x: x.sum(), axis=1) > 0 
        self.y = df[(df[self.y_col].notna()) & (~self.nan_lines)][self.y_col]
        self.x = df[(df[self.y_col].notna()) & (~self.nan_lines)].drop(self.y_col, axis=1)
    def split_train_test(self, include_nan=True, test_size=0.2):
        # Divide between train and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=1)
        
        # Convert to DataFrame
        self.x_train = pd.DataFrame(self.x_train, columns=self.x_cols)
        self.x_test = pd.DataFrame(self.x_test, columns=self.x_cols)
        # Treate Nan
        if include_nan:
            self.x_train = pd.concat([self.df.loc[self.nan_lines, self.x_cols], self.x_train], axis=0)
            self.y_train = pd.concat([self.df.loc[self.nan_lines, self.y_col], self.y_train])

        ## Reset index
        self.x_train = self.x_train.reset_index(drop=True)
        self.x_test = self.x_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        return {
            "train_shape": self.x_train.shape[0],
            "test_shape": self.x_test.shape[0],
            "train_val_count":  self.y_train.value_counts().to_dict(),
            "test_val_count":  self.y_test.value_counts().to_dict()

        }

    def x_normalizator(self, num_normalizers = []):
        self.logs[f"train_normalization"] = {}
        self.logs[f"test_normalization"] = {}

        # Loop for each Normalizer
        for i, norm in enumerate(num_normalizers):
            # Create model intance
            self.models[f'scaled_{i}'] = dict()

            # Fit with training data
            normalizor = norm.fit(self.x_train.loc[:, self.num_cols])
            
            # Scale numeric data
            x_num_train_scaled = normalizor.transform(self.x_train.loc[:, self.num_cols])
            x_num_test_scaled = normalizor.transform(self.x_test.loc[:, self.num_cols])
            x_num_train_scaled = pd.DataFrame(x_num_train_scaled, columns=self.num_cols)
            x_num_test_scaled = pd.DataFrame(x_num_test_scaled, columns=self.num_cols)


            # One Hot Encode Categoricol Data
            if len(self.cat_cols) > 0:
                # Apply get dummies to One Hot Encode
                x_cat_train_oh = pd.get_dummies(self.x_train[self.cat_cols], columns= self.cat_cols, drop_first=True)
                x_cat_test_oh = pd.get_dummies(self.x_test[self.cat_cols], columns= self.cat_cols, drop_first=True)

                # Find nan columns to replace    
                for new_col in x_cat_train_oh.columns:
                    cat_col = new_col.split('_')[0]
                    x_cat_train_oh.loc[self.x_train[cat_col].isna(), new_col] = np.nan


            self.models[f'scaled_{i}']['x_train'] = pd.concat([x_cat_train_oh, x_num_train_scaled], axis=1)
            self.models[f'scaled_{i}']['x_test'] = pd.concat([x_cat_test_oh, x_num_test_scaled], axis=1)
            
            self.logs["train_normalization"][f'scaled_{i}'] = self.models[f'scaled_{i}']['x_train'].describe().to_dict()
            self.logs["test_normalization"][f'scaled_{i}'] = self.models[f'scaled_{i}']['x_test'].describe().to_dict()

        return self.logs["train_normalization"], self.logs["test_normalization"]
          
    def inpututation(self, inputers):
        for key in self.models.keys():
            for i, inputer in enumerate(inputers):
                train_cols = self.models[key]['x_train'].columns
                inputed_data = inputer.fit_transform(self.models[key]['x_train'])
                self.models[key][f'inputed_{i}_x_train'] = pd.DataFrame(inputed_data, columns=train_cols)
    
    def _generate_model_path(self, model_name):
        path = f'{self.output_path}/{model_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _gen_grid_search(self, _x_train, _x_test, _y_train, _y_test, g_params, model, title):
        best_model_grid = {}
        for sc in g_params["scoring"]:
            grid_search = GridSearchCV(
                    estimator=model, 
                    scoring = sc['metric'], 
                    cv = g_params["cv"], 
                    param_grid=g_params["param_grid"]
            )
            grid_search.fit(_x_train, _y_train)
            train_metric = grid_search.score(_x_train, _y_train)
            test_metric = grid_search.score(_x_test, _y_test)
            _y_pred = grid_search.predict(_x_test)
            best_model_grid[f"{sc['name']}"] = grid_search.best_estimator_
            path = self._generate_model_path(title)
            self._plot_confusion_matrix(path, _y_test, _y_pred, grid_search.classes_, sc['name'])
            self._generate_metric_csv(path, _y_pred, _y_test, train_metric, test_metric, grid_search.best_params_, sc['name'])

        return best_model_grid


    
    def _plot_confusion_matrix(self, path, _y_test, _y_pred, classes, metric):
        cm = confusion_matrix(_y_test, _y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=classes)
        disp.plot(cmap="Blues")
        plt.savefig(path + f"/confusion_matrix_{metric}.jpg")
        
    def _generate_metric_csv(self, path, _y_pred, _y_test, train_metric, test_metric, best_paramns, metric):
        metrics = [
                ["train", train_metric],
                ["test", test_metric],
                ["accuracy", accuracy_score(_y_test, _y_pred)],
                ["f1-macro", f1_score(_y_test, _y_pred, average='macro')],
                ["precision-macro", precision_score(_y_test, _y_pred, average='macro')],
                ["recall-macro", precision_score(_y_test, _y_pred, average='macro')],
                ["f1-micro", f1_score(_y_test, _y_pred, average='micro')],
                ["precision-micro", precision_score(_y_test, _y_pred, average='micro')],
                ["recall-micro", precision_score(_y_test, _y_pred, average='micro')],
                ["f1-weighted", f1_score(_y_test, _y_pred, average='weighted')],
                ["precision-weighted", precision_score(_y_test, _y_pred, average='weighted')],
                ["recall-weighted", precision_score(_y_test, _y_pred, average='weighted')],
                ["best_paramns", best_paramns]
        ]
        metrics_df = pd.DataFrame(metrics, columns=["metric", "value"])
        metrics_df.to_csv(path + f"/scores_{metric}.csv", index=None)

    # def _gen_random_search(self, _x_train, _x_test, _y_train, _y_test, r_params, model, title):

    
    # def _gen_single_model(self, _x_train, _x_test, _y_train, _y_test, model, title):


    def _model_selection_router(self, ev_mod, tt_set, model_name):

        if "selection" in ev_mod:
            if ev_mod["selection"] == "grid":
                return self._gen_grid_search(*tt_set, ev_mod["r/g"], ev_mod["model"], model_name)
            # elif ev_mod["selection"] == "random":
                # return self._gen_random_search(*tt_set, ev_mod["r/g"], ev_mod["model"], ev_mod["title"])
        # else:
        #     return self._gen_single_model(*tt_set, ev_mod["model"], ev_mod["title"])
                


    def modelEvaluator(self, evaluate_models: list, scaled=False, inputed=False):
        if inputed and scaled:
            best_models = {}
            for scaled_model in self.models.keys():
                inputations = [inp for inp in self.models[scaled_model].keys() if 'inputed' in inp]
                for inputed_model in inputations:
                    for ev_mod in evaluate_models:
                        train_test_set = [
                            self.models[scaled_model][inputed_model],
                            self.models[scaled_model]["x_test"],
                            self.y_train,
                            self.y_test
                        ]
                        model_name = ev_mod["title"] + f"_{scaled_model}_" + f"{inputed_model}"
                       
                        best_models[model_name] = self._model_selection_router(ev_mod, train_test_set, model_name)
                return best_models
            
        elif scaled:
            best_models = {}
            for scaled_model in self.models.keys():
                for ev_mod in evaluate_models:
                    train_test_set = [
                            self.models[scaled_model]["x_train"],
                            self.models[scaled_model]["x_test"],
                            self.y_train,
                            self.y_test
                    ]
                    model_name = ev_mod["title"] + f"_{scaled_model}"
                    best_models[model_name] = self._model_selection_router(ev_mod, train_test_set, model_name)
            return best_models
        else: 
            best_models = {}
            for ev_mod in evaluate_models:
                train_test_set = [
                    self.x_train,
                    self.x_test,
                    self.y_train,
                    self.y_test
                ]
                model_name = ev_mod["title"]
                best_models[model_name] = self._model_selection_router(ev_mod, train_test_set)
         
            return best_models
