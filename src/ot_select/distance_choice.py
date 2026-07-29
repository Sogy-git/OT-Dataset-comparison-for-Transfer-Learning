import torch
import torch.nn as nn
import ot
import numpy as np
from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import cdist
from encoder import extract_embeddings
from data_choice import create_data_loaders
from model import SimpleCNN
from train_func import train, test, trainloop

def KNN_distance_choice(embeddings_train, embeddings_test, k=5, subset_size=5000):
    # Calculate pairwise distances between training and test embeddings
    distances = torch.cdist(embeddings_test, embeddings_train)

    # Get the indices of the k nearest neighbors for each test sample
    _, knn_indices = torch.topk(distances, k=k, largest=False)

    # Extract the unique indices of the k nearest neighbors across all test samples
    unique_knn_indices = torch.unique(knn_indices)

    if subset_size == None:
        return embeddings_train[unique_knn_indices], unique_knn_indices
    
    else:
        # Limit the selection to the subset_size candidates closest to any test embedding
        candidate_distances = distances[:, unique_knn_indices].min(dim=0).values

        selected_positions = torch.topk(
            candidate_distances,
            k=min(subset_size, len(unique_knn_indices)),
            largest=False
        ).indices
    unique_knn_indices = unique_knn_indices[selected_positions]

    # Return the unique nearest neighbour vectors from the training embeddings
    return embeddings_train[unique_knn_indices], unique_knn_indices


def OT_distance_choice(embeddings_train, embeddings_test, subset_size=1000):

    # Calculate the candidates for the optimal transport plan through KNN distance choice
    candidates, candidate_indices = KNN_distance_choice(
        embeddings_train,
        embeddings_test,
        k=150,
        subset_size=None
    )

    # TEMPORARY FIX
    candidates = candidates.detach().cpu().numpy()
    candidate_indices = candidate_indices.detach().cpu().numpy()

    # TEMPORARY FIX
    numpy_embeddings_test = embeddings_test.detach().cpu().numpy()
    numpy_embeddings_train = embeddings_train.detach().cpu().numpy()

    # Calculate the cost matrix between candidate training embeddings
    # and test embeddings
    cost_matrix = cdist(
        candidates,
        numpy_embeddings_test,
        metric='euclidean'
    )

    source_weights = np.full(
        len(candidates),
        1.0 / subset_size
    )

    # Target weights for the test embeddings (uniform distribution)
    target_weights = np.full(
        len(embeddings_test),
        1.0 / len(embeddings_test)
    )

    # Compute the optimal transport plan using Wasserstein distance
    transport_plan = ot.partial.partial_wasserstein(
        source_weights,
        target_weights,
        cost_matrix,
        m=1.0,
        nb_dummies=20
    )

    # Total mass sent by every candidate
    transported_mass = transport_plan.sum(axis=1)

    nearest_target_cost = cost_matrix.min(axis=1)

    candidate_order = np.lexsort(
        (nearest_target_cost, -transported_mass)
    )

    chosen_candidate_positions = candidate_order[:subset_size]
    selected_indices = candidate_indices[chosen_candidate_positions]

    return selected_indices

def OTDD_distance_choice(embeddings_train, labels_train, embeddings_target, labels_target, subset_size=5000):

    # Calculate the candidates for the optimal transport plan
    candidates, candidate_indices = KNN_distance_choice(
        embeddings_train,
        embeddings_target,
        k=150,
        subset_size=None
    )

    # Convert tensors to NumPy arrays
    candidates = candidates.detach().cpu().numpy()
    candidate_indices = candidate_indices.detach().cpu().numpy()

    numpy_embeddings_train = embeddings_train.detach().cpu().numpy()
    numpy_embeddings_test = embeddings_target.detach().cpu().numpy()

    numpy_labels_train = (
        labels_train.numpy() if torch.is_tensor(labels_train) else labels_train
    )
    numpy_labels_test = (
        labels_target.numpy() if torch.is_tensor(labels_target) else labels_target
    )

    # Get the labels of the candidate training examples
    candidate_labels = numpy_labels_train[candidate_indices]

    # Find the unique source and target classes
    source_classes = np.unique(numpy_labels_train)
    target_classes = np.unique(numpy_labels_test)

    # Create a matrix to store the distance between each pair
    # of source and target class distributions
    class_costs = np.zeros((
        int(source_classes.max()) + 1,
        int(target_classes.max()) + 1
    ))

    # Calculate the Wasserstein distance between every source
    # class distribution and every target class distribution
    for source_class in source_classes:
        source_class_embeddings = numpy_embeddings_train[
            numpy_labels_train == source_class
        ]

        source_class_weights = np.full(
            len(source_class_embeddings),
            1.0 / len(source_class_embeddings)
        )

        for target_class in target_classes:
            target_class_embeddings = numpy_embeddings_test[
                numpy_labels_test == target_class
            ]

            target_class_weights = np.full(
                len(target_class_embeddings),
                1.0 / len(target_class_embeddings)
            )

            class_cost_matrix = cdist(
                source_class_embeddings,
                target_class_embeddings,
                metric="sqeuclidean"
            )

            class_costs[
                int(source_class),
                int(target_class)
            ] = ot.emd2(
                source_class_weights,
                target_class_weights,
                class_cost_matrix
            )

    # Calculate the squared Euclidean distance between individual
    # candidate and target embeddings
    feature_costs = cdist(
        candidates,
        numpy_embeddings_test,
        metric="sqeuclidean"
    )

    # Assign the appropriate class-distribution distance to every
    # candidate-target pair
    label_costs = class_costs[
        candidate_labels[:, None],
        numpy_labels_test[None, :]
    ]

    # Combine individual feature distance and class distance
    cost_matrix = feature_costs + label_costs

    # Source weights for the candidate training embeddings
    source_weights = np.full(
        len(candidates),
        1.0 / subset_size
    )

    # Target weights for the target embeddings
    target_weights = np.full(
        len(numpy_embeddings_test),
        1.0 / len(numpy_embeddings_test)
    )

    # Compute the partial optimal transport plan
    transport_plan = ot.partial.partial_wasserstein(
        source_weights,
        target_weights,
        cost_matrix,
        m=1.0,
        nb_dummies=20
    )

    # Calculate the total mass transported by each candidate
    transported_mass = transport_plan.sum(axis=1)

    # Use the closest combined feature and label cost to break ties
    nearest_target_cost = cost_matrix.min(axis=1)

    candidate_order = np.lexsort(
        (nearest_target_cost, -transported_mass)
    )

    # Select the candidates receiving the most transport mass
    chosen_candidate_positions = candidate_order[:subset_size]

    # Convert candidate positions back to original training indices
    selected_indices = candidate_indices[
        chosen_candidate_positions
    ]

    return selected_indices