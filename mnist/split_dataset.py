import wandb
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split


def main():
    with wandb.init(job_type="preprocess-data") as run:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)
        datasets = [(x_train, y_train), (x_val, y_val), (x_test, y_test)]
        names = ("train", "validation", "test")
        dataset_artifact = wandb.Artifact(
            "mnist-data",
            "dataset",
            description=f"MNIST data loaded via tensorflow {tf.__version__}",
            metadata={
                "train_examples": len(y_train),
                "validation_examples": len(y_val),
                "test_examples": len(y_test)
            }
        )
        for name, ds in zip(names, datasets):
            fname = f"{name}.npz"
            np.savez(fname, x=ds[0], y=ds[1])
            dataset_artifact.add_file(fname, name=name)
        run.log_artifact(dataset_artifact)

if __name__ == "__main__":
    main()
