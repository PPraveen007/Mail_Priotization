import pickle
from sklearn.metrics import precision_recall_curve
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def generate_report(y_test, y_pred, title):
    print(f"For {title} -----------------------------------")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:",  round(accuracy, 2))

    print("Other Metrics:")
    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

    classes = np.unique(y_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="y_pred labels", ylabel="True labels", xticklabels=classes,
           yticklabels=classes, title=f"Confusion matrix: {title}")
    plt.yticks(rotation=0)
    print(f"----------------------------------------------")


def save_model(model, file_name: str = "saved_model"):
    pickle.dump(model, open(file_name, 'wb'))
