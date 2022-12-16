import tensorflow as tf
import flwr as fl

import pandas as pd
import numpy as np
from data import load_data
from model import Model


import argparse

# sys.path.insert(0, "../data/store")

x_train, y_train, x_test, y_test = load_data()
# x_train, y_train, x_test, y_test = None, None, None, None

model = Model()


class StoreClient(fl.client.NumPyClient):
    def __init__(
        self,
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    ) -> None:
        super().__init__()
        self.model = model.model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.y_train, epochs=10)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "mean_squared_error": history.history["mean_squared_error"][0],
            # "val_loss": history.history["val_loss"][0],
            # "val_mean_squared_error": history.history["val_mean_squared_error"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mean_squared_error, mean_absolute_error = self.model.evaluate(
            self.x_test, self.y_test
        )
        return (
            loss,
            len(self.x_test),
            {
                "mean_squared_error": float(mean_squared_error),
                "mean_absolute_error": mean_absolute_error,
            },
        )


# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=StoreClient())


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client", type=int, choices=range(1, 46), required=True)
    args = parser.parse_args()
    print(str(args.client))

    x_train, y_train, x_test, y_test = load_data(str(args.client))

    model = Model()

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=StoreClient(model, x_train, y_train, x_test, y_test),
    )


if __name__ == "__main__":
    main()
