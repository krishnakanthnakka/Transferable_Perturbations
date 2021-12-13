from .imagenet import imagenet_evaluation, imagenet_evaluation_adv
from cda.data.datasets import ImageDataset


def evaluate(dataset, gt_labels, predictions_clean, output_dir=None, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(gt_labels=gt_labels, predictions_clean=predictions_clean, **kwargs,)

    if isinstance(dataset, ImageDataset):
        return imagenet_evaluation(**args)

    raise NotImplementedError


def evaluate_adv(dataset, gt_labels, predictions_adv, predictions_clean=None, output_dir=None, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(gt_labels=gt_labels, predictions_clean=predictions_clean, predictions_adv=predictions_adv,
                **kwargs,)

    if isinstance(dataset, ImageDataset):
        return imagenet_evaluation_adv(**args)

    raise NotImplementedError
