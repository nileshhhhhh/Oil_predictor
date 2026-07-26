import pickle
from pathlib import Path


def save_model(model, model_path):
    """
    Persist a fitted estimator (or sklearn Pipeline) to disk as a .pkl file.

    Parameters
    ----------
    model : fitted estimator
    model_path : str or Path, e.g. '../data/best_model.pkl'

    Returns
    -------
    Path to the file that was written.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


def load_model(model_path):
    """
    Load a previously pickled estimator back into memory.

    Parameters
    ----------
    model_path : str or Path to a .pkl file written by save_model()

    Returns
    -------
    The unpickled estimator, ready to call .predict() on.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No model file found at: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
