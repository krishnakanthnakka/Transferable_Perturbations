import logging
import numpy as np


def imagenet_evaluation_adv(gt_labels, predictions_adv, predictions_clean, iteration=None):

    logger = logging.getLogger("CDA.inference")
    accuracy = np.sum(gt_labels == predictions_adv)
    accuracy /= len(gt_labels)
    metrics = {'Adversarial_Accuracy': accuracy}
    logger.info("Adversarial Accuracy: {:.2f}%".format(100 * metrics['Adversarial_Accuracy']))

    if predictions_clean is not None:
        fooling = np.sum(predictions_clean !=
                         predictions_adv) / len(predictions_adv)
        metrics.update({'Fooling': fooling})
        logger.info("Fooling Rate: {:.2f}%".format(100 * metrics['Fooling']))

        clean_accuracy = np.sum(gt_labels == predictions_clean)
        clean_accuracy /= len(gt_labels)
        metrics.update({'Normal_Accuracy': clean_accuracy})
        logger.info("Normal Accuracy: {:.2f}%".format(100 * metrics['Normal_Accuracy']))

        error_rate_improvement = metrics['Normal_Accuracy'] - metrics['Adversarial_Accuracy']
        metrics.update({'Error_rate_improvement': error_rate_improvement})
        logger.info("Error Rate Improvement: {:.2f}%".format(100 * metrics['Error_rate_improvement']))

    return dict(metrics=metrics)


def imagenet_evaluation(gt_labels, predictions_clean=None, iteration=None):

    logger = logging.getLogger("CDA.inference")
    accuracy = np.sum(gt_labels == predictions_clean)
    accuracy /= len(gt_labels)
    metrics = {'Normal_Accuracy': accuracy}
    logger.info("Normal Accuracy: {:.2f}%".format(100 * metrics['Normal_Accuracy']))
    return dict(metrics=metrics)
