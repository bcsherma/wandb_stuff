import wandb
import numpy as np
import tensorflow as tf
import argparse

TABLE_COLUMNS = ["image", "label", "prediction"]
TABLE_COLUMNS.extend([f"score_{i}" for i in range(10)])

def build_pred_table(dataset, logits):
    data = [
        [wandb.Image(dataset["x"][idx]), dataset["y"][idx], np.argmax(logits[idx]), *logits[idx]]
        for idx in range(len(logits))
    ]
    table = wandb.Table(data=data, columns=TABLE_COLUMNS)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model artifact to be evaluated")
    parser.add_argument("dataset", help="dataset to evaluate model on")
    args = parser.parse_args()
    with wandb.init(job_type="evaluate") as run:
        model_artifact = run.use_artifact(args.model)
        run.config.update(model_artifact.metadata)
        data_artifact = run.use_artifact(args.dataset)
        model = tf.keras.models.load_model(model_artifact.get_path("model").download())
        test_data = np.load(data_artifact.get_path("test").download())
        logits = model.predict(test_data["x"])
        preds = np.argmax(logits, axis=1)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(test_data["y"], logits).numpy()
        accuracy = tf.keras.metrics.Accuracy()(test_data["y"], preds).numpy()
        run.summary["loss"] = loss
        run.summary["accuracy"] = accuracy
        run.log({"predictions": build_pred_table(test_data, logits)})



if __name__ == "__main__":
    main()
