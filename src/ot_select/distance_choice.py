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

mnist_train_loader, mnist_test_loader, semeion_train_loader, semeion_test_loader = create_data_loaders("full", batch_size=10, limit=1000)

embeddings_train, embeddings_test = extract_embeddings(type="full", batch_size=10, limit=1000, train_loader=mnist_train_loader, target_loader=semeion_train_loader)

def KNN_distance_choice(embeddings_train, embeddings_test, k=5):
    # Calculate pairwise distances between training and test embeddings
    distances = torch.cdist(embeddings_test, embeddings_train)

    # Get the indices of the k nearest neighbors for each test sample
    _, knn_indices = torch.topk(distances, k=k, largest=False)

    # Extract the unique indices of the k nearest neighbors across all test samples
    unique_knn_indices = torch.unique(knn_indices)

    # Limit the selection to the 5000 candidates closest to any test embedding
    candidate_distances = distances[:, unique_knn_indices].min(dim=0).values
    selected_positions = torch.topk(
        candidate_distances,
        k=min(5000, len(unique_knn_indices)),
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
        k=150
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

# Use KNN distance choice to select training samples based on embeddings
KNN_selected_vectors, KNN_selected_indices = KNN_distance_choice(embeddings_train, embeddings_test, k=150)
KNN_train = torch.utils.data.Subset(mnist_train_loader.dataset, KNN_selected_indices)
KNN_loader = torch.utils.data.DataLoader(KNN_train, batch_size=10, shuffle=True)

# Train a new model on the selected KNN training samples
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
trainloop("KNN_test", device, lr=0.001, epoch=10, train_loader=KNN_loader, test_loader=semeion_test_loader)

# Use OT distance choice to select training samples based on embeddings
OT_selected_indices = OT_distance_choice(embeddings_train, embeddings_test, subset_size=5000)
OT_train = torch.utils.data.Subset(mnist_train_loader.dataset, OT_selected_indices)
OT_loader = torch.utils.data.DataLoader(OT_train, batch_size=10, shuffle=True)

# Train a new model on the selected OT training samples
trainloop("OT_test", device, lr=0.001, epoch=10, train_loader=OT_loader, test_loader=semeion_test_loader)
