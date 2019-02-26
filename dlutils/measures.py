# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from sklearn.metrics import roc_auc_score


def f1_from_pr(precision, recall):
    if precision == 0.0 or recall == 0.0:
        return 0
    return 2.0 * precision * recall / (precision + recall)


def f1_from_tp_fp_fn(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return f1_from_pr(precision, recall)


def openset_f1(label_inlier, prediction_inlier, threshold, correctly_classified):
    assert label_inlier.shape == prediction_inlier.shape
    assert label_inlier.shape == correctly_classified.shape
    y = np.greater(prediction_inlier, threshold)
    not_y = np.logical_not(y)
    label_outlier = np.logical_not(label_inlier)

    correctly_predicted_as_inlier = np.logical_and(y, label_inlier)
    correctly_predicted_as_outlier = np.logical_and(not_y, label_outlier)

    correct = np.logical_or(np.logical_and(correctly_predicted_as_inlier, correctly_classified), correctly_predicted_as_outlier)
    not_correct = np.logical_not(correct)

    true_positive = np.sum(correct)
    false_positive = np.sum(np.logical_and(not_correct, label_outlier))
    false_negative = np.sum(np.logical_and(not_correct, label_inlier))

    if true_positive + false_positive > 0:
        return f1_from_tp_fp_fn(true_positive, false_positive, false_negative)
    else:
        return 0


def f1(label, prediction, threshold):
    assert label.shape == prediction.shape
    y = np.greater(prediction, threshold)
    not_label = np.logical_not(label)

    correct = y == label
    not_correct = np.logical_not(correct)

    true_positive = np.sum(correct)
    false_positive = np.sum(np.logical_and(not_correct, label))
    false_negative = np.sum(np.logical_and(not_correct, not_label))

    if true_positive + false_positive > 0:
        return f1_from_tp_fp_fn(true_positive, false_positive, false_negative)
    else:
        return 0


def auc(label, prediction):
    try:
        return roc_auc_score(label, prediction)
    except ValueError:
        return 0


if __name__ == '__main__':
    threshold = 0.5
    label = np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
    prediction = np.asarray([0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.7, 0.7, 0.1, 0.1])

    print(f1(label, prediction, threshold))

    print(openset_f1(label, prediction, threshold, np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 1, 1])))

    print(auc(np.asarray([0, 0, 0]), np.asarray([0.1, 0.1, 0.1])))
