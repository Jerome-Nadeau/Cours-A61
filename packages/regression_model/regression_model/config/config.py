import pathlib

import regression_model

import pandas as pd

#PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
#PACKAGE_ROOT = "C:/Users/jerom/OneDrive/Documents/420-A61-SF/Cours-A61/packages/regression_model"
import os

current_file_path = os.path.abspath(regression_model.__file__)
PACKAGE_ROOT = os.path.dirname(current_file_path)
#print(parent_directory)

#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
TRAINED_MODEL_DIR = '/'.join([ PACKAGE_ROOT , "trained_models"])
#TRAINED_MODEL_DIR = "C:/Users/jerom/OneDrive/Documents/420-A61-SF/Cours-A61/packages/regression_model/regression_model/trained_models"
#DATASET_DIR = PACKAGE_ROOT / "datasets"
DATASET_DIR = '/'.join([ PACKAGE_ROOT , "datasets"])
#DATASET_DIR = "C:/Users/jerom/OneDrive/Documents/420-A61-SF/Cours-A61/packages/regression_model/regression_model/datasets"


# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = "SalePrice"


# variables
FEATURES = [
    "MSSubClass",
    "MSZoning",
    "Neighborhood",
    "OverallQual",
    "OverallCond",
    "YearRemodAdd",
    "RoofStyle",
    "MasVnrType",
    "BsmtQual",
    "BsmtExposure",
    "HeatingQC",
    "CentralAir",
    "1stFlrSF",
    "GrLivArea",
    "BsmtFullBath",
    "KitchenQual",
    "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageCars",
    "PavedDrive",
    "LotFrontage",
    # this one is only to calculate temporal variable:
    "YrSold",
]

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = "YrSold"

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ["LotFrontage"]

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = [
    "MasVnrType",
    "BsmtQual",
    "BsmtExposure",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
]

TEMPORAL_VARS = "YearRemodAdd"

# variables to log transform
NUMERICALS_LOG_VARS = ["LotFrontage", "1stFlrSF", "GrLivArea"]

# categorical variables to encode
CATEGORICAL_VARS = [
    "MSZoning",
    "Neighborhood",
    "RoofStyle",
    "MasVnrType",
    "BsmtQual",
    "BsmtExposure",
    "HeatingQC",
    "CentralAir",
    "KitchenQual",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
]

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]


PIPELINE_NAME = "lasso_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
