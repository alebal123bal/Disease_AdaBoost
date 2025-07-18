"""
This script is used to train a classifier using the provided training data.
"""

import time
from preprocess.dataset_preprocess import load_matrix_weigths_labels
from adaboost_smart.adaboost import AdaBoost, ClassifierScoreCheck

if __name__ == "__main__":
    # Load the dataset
    FILE_PATH = "./dataset/pima_indians_diabetes.csv"  # Replace with your dataset path
    FEATURE_EVAL_MATRIX, SAMPLE_WEIGHTS, SAMPLE_LABELS = load_matrix_weigths_labels(
        FILE_PATH,
        bias_factor=300.0,  # More weight to positive samples for better classification performance
    )

    # Initialize and train the AdaBoost classifier
    adaboost_classifier = AdaBoost(
        feature_eval_matrix=FEATURE_EVAL_MATRIX,
        sample_weights=SAMPLE_WEIGHTS,
        sample_labels=SAMPLE_LABELS,
        n_stages=4,
        aggressivness=0.15,
        feature_per_stage=7,
    )

    print("\n🔄 Training AdaBoost Classifier...\n")

    start_time = time.time()

    adaboost_classifier.train()

    print("\n🎯 Training completed.\n")

    print(f"\n⏱️ Total training time: {(time.time() - start_time)} seconds")

    my_classifier = ClassifierScoreCheck(
        feature_eval_matrix=FEATURE_EVAL_MATRIX,
        sample_labels=SAMPLE_LABELS,
    )

    my_classifier.analyze()
