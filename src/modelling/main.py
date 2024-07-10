# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
import pickle
import pandas as pd
from preprocessing import preprocessing
from training import train 
from loguru import logger


filepath = 'src/web_service/local_objects'
PATH = '../data/abalone.csv'
def main(trainset_path: PATH) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    data = pd.read_csv(trainset_path)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessing(data)

    # (Optional) Pickle encoder if need be

    # Train model
    model = train(X_train, y_train)

    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder
    with open(filepath, "wb") as f:
        pickle.dump(object, f)
    logger.info(f"Pickled model to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
