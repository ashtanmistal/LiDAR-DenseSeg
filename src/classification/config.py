import ml_collections
import multiprocessing
def get_config():
    """
    Get configuration parameters for dataset, models, and training.
    :return: configuration parameters
    """

    config = ml_collections.ConfigDict()

    # Hyperparameters for dataset.
    config.lidar_directory = "../../data/las/"
    config.pkl_classified_directory = "../../data/pkl/classified/"
    config.pkl_unclassified_directory = "../../data/pkl/unclassified/"
    config.ratio_tr_data = 0.8  # ratio of training data to total data
    config.num_classes = 17
    config.num_workers = multiprocessing.cpu_count()

    # TODO add the rest of the hyperparameters for dataset, models, and training.

    return config


