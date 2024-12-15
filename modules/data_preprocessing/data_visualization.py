import os
from collections import Counter

import matplotlib.pyplot as plt


def show_labels_count(labels):
    return Counter(labels)


def show_labels_chart(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color="skyblue", edgecolor="black")

    plt.xticks(classes, labels=classes, fontsize=12)

    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Size of classes", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def generate_training_plot(history, model_name, save_location):
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))

    axs[0].plot(history["accuracy"], color="r", label="accuracy")
    axs[0].plot(history["val_accuracy"], color="b", label="val_accuracy")
    axs[1].plot(history["loss"], color="r", label="loss")
    axs[1].plot(history["val_loss"], color="b", label="val_loss")

    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlabel("Epchos")
    axs[0].set_ylabel("Accuracy Score")
    axs[1].set_xlabel("Epchos")
    axs[1].set_ylabel("Loss")

    save_path = os.path.join(save_location, f"{model_name}_training_chart.png")

    plt.savefig(save_path)
    plt.close(fig)
