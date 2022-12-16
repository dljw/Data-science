from typing import Dict, Optional, Tuple
import flwr as fl

from data import load_data
from model import Model


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = Model().model

    # Create strategy
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.3,
    #     fraction_evaluate=0.2,
    #     min_fit_clients=3,
    #     min_evaluate_clients=2,
    #     min_available_clients=10,
    #     evaluate_fn=get_evaluate_fn(model),
    #     on_fit_config_fn=fit_config,
    #     on_evaluate_config_fn=evaluate_config,
    #     initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    # )

    strategy = fl.server.strategy.FedAdagrad(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=10,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    model.save("../models/federated_model_adagrad.pkt")
    print("Model saved")
    # # Start Flower server (SSL-enabled) for four rounds of federated learning
    # fl.server.start_server(
    #     server_address="0.0.0.0:8080",
    #     config=fl.server.ServerConfig(num_rounds=4),
    #     strategy=strategy,
    #     certificates=(
    #         Path(".cache/certificates/ca.crt").read_bytes(),
    #         Path(".cache/certificates/server.pem").read_bytes(),
    #         Path(".cache/certificates/server.key").read_bytes(),
    #     ),
    # )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    x_train, y_train, x_val, y_val = load_data()

    # Use the last 5k training examples as a validation set

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters

        loss, mean_squared_error, mean_absolute_error = model.evaluate(x_val, y_val)
        return loss, {
            "mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
        }

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
