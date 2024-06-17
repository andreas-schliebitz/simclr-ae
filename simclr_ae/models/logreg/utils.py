from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score


def create_metrics(
    num_classes: int, task: str = "multiclass", average: str = "macro"
) -> MetricCollection:
    metric_params = {
        "num_classes": num_classes,
        "task": task,
        "average": average,
    }

    match average:
        case "micro":
            metrics = [Accuracy(**metric_params)]
        case "macro":
            metrics = [
                Accuracy(**metric_params),
                Precision(**metric_params),
                Recall(**metric_params),
                F1Score(**metric_params),
            ]
        case _:
            raise ValueError(f"Unsupported averaging method: '{average}'")

    return MetricCollection(metrics)
