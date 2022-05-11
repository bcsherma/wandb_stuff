import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

DEFAULT_CONFIG = {
    "layer_1": 512,
    "activation_1": "relu",
    "dropout": 0.25,
    "layer_2": 10,
    "activation_2": "softmax",
    "optimizer": "sgd",
    "loss": "sparse_categorical_crossentropy",
    "metric": "accuracy",
    "epoch": 6,
    "batch_size": 256,
}


def build_model(config):
    # Build a model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(config.layer_1, activation=config.activation_1),
            tf.keras.layers.Dropout(config.dropout),
            tf.keras.layers.Dense(config.layer_2, activation=config.activation_2),
        ]
    )
    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])
    return model


def main():
    with wandb.init(config=DEFAULT_CONFIG, job_type="train") as run:
        data_artifact = run.use_artifact("mnist-data:latest")
        train_data = np.load(data_artifact.get_path("train").download())
        val_data = np.load(data_artifact.get_path("validation").download())
        model = build_model(run.config)
        logging_callback = WandbCallback(save_model=False)
        model.fit(
            x=train_data["x"],
            y=train_data["y"],
            epochs=run.config.epoch,
            batch_size=run.config.batch_size,
            validation_data=(val_data["x"], val_data["y"]),
            callbacks=[logging_callback, tf.keras.callbacks.TensorBoard()],
        )
        model.save("model.keras")
        model_artifact = wandb.Artifact("mnist-model", "model", metadata=dict(run.config))
        model_artifact.add_file("model.keras", name="model")
        run.log_artifact(model_artifact)


if __name__ == "__main__":
    main()
