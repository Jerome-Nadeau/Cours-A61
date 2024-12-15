from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from regression_model.config import config
from regression_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv('/'.join([ f"{config.DATASET_DIR}" , f"{file_name}"]))
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = '/'.join([ f"{config.TRAINED_MODEL_DIR}" , f"{save_file_name}"])

    remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    file_path =  '/'.join([ f"{config.TRAINED_MODEL_DIR}" , f"{file_name}"])
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    trained_model_dir = Path(config.TRAINED_MODEL_DIR)
    for model_file in trained_model_dir.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()
