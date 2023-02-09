from sklearn.metrics import classification_report


class ClassificationReport():
    name = None

    def __init__(self, y_true, y_pred, labels, name) -> None:
        self.name = name
        self.cr = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)

    def save(self):
        f = open(f'{self.name}.txt', 'w+')
        f.write(self.cr)
        f.close