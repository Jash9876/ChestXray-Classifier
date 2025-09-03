import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_gen):
    test_gen.reset()
    pred = model.predict(test_gen)
    pred_classes = np.argmax(pred, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print("Classification Report:")
    print(classification_report(true_classes, pred_classes, target_names=class_labels))

    cm = confusion_matrix(true_classes, pred_classes)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("saved_model/confusion_matrix.png")
    plt.show()
