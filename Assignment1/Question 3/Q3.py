import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def load_graph(filename):
    in_degree = {}
    out_degree = {}
    weights = {}
    neighbors = {}

    with open(filename, "r") as file:
        for line in file:
            src, dest, weight = map(int, line.strip().split(",")[:3])

            if src not in out_degree:
                out_degree[src] = 0
            if dest not in in_degree:
                in_degree[dest] = 0

            if src not in weights:
                weights[src] = {}

            if src not in neighbors:
                neighbors[src] = set()
            if dest not in neighbors:
                neighbors[dest] = set()

            out_degree[src] += weight
            in_degree[dest] += weight

            weights[src][dest] = weight

            neighbors[src].add(dest)
            neighbors[dest].add(src)

    return in_degree, out_degree, weights, neighbors


def compute_total_degree(in_degree, out_degree):
    total_degree = {}
    all_nodes = set(in_degree.keys()).union(out_degree.keys())

    for node in all_nodes:
        in_val = in_degree[node] if node in in_degree else 0
        out_val = out_degree[node] if node in out_degree else 0
        total_degree[node] = in_val + out_val

    return total_degree


def plot_degree_distribution(total_degree):
    degree_counts = Counter(total_degree.values())
    k_vals = np.array(list(degree_counts.keys()))
    p_k = np.array(list(degree_counts.values())) / sum(degree_counts.values())

    plt.figure(figsize=(8, 6))
    plt.scatter(k_vals, p_k)
    plt.xlabel("Weighted Degree k")
    plt.ylabel("P(k)")
    plt.title("Weighted Degree Distribution")
    plt.grid(True)
    plt.show()


def compute_weighted_clustering(total_degree, weights, neighbors):
    clustering_coeffs = {}

    for node in total_degree:
        if node not in neighbors:
            clustering_coeffs[node] = 0.0
            continue

        k = len(neighbors[node])

        if k < 2:
            clustering_coeffs[node] = 0.0
            continue

        sum_weights = 0
        neighbors_list = list(neighbors[node])

        for i in range(k):
            for j in range(i + 1, k):
                node_i = neighbors_list[i]
                node_j = neighbors_list[j]

                if node_i in weights and node_j in weights[node_i]:
                    weight_ij = (
                        weights[node].get(node_i, 0) +
                        weights[node].get(node_j, 0)
                    ) / 2
                    sum_weights += weight_ij

        clustering_coeffs[node] = (2 * sum_weights) / (k * (k - 1))

    return clustering_coeffs


def plot_clustering_vs_degree(total_degree, clustering_coeffs):
    degrees = list(total_degree.values())
    clustering_values = list(clustering_coeffs.values())

    plt.figure(figsize=(8, 6))
    plt.scatter(degrees, clustering_values, alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree k")
    plt.ylabel("Clustering Coefficient")
    plt.title("Clustering Coefficient vs Degree")
    plt.grid(True, which="both")
    plt.show()


def main():
    filename = "soc-sign-bitcoinotc.csv"

    in_degree, out_degree, weights, neighbors = load_graph(filename)
    total_degree = compute_total_degree(in_degree, out_degree)

    plot_degree_distribution(total_degree)

    clustering_coeffs = compute_weighted_clustering(
        total_degree, weights, neighbors
    )

    plot_clustering_vs_degree(total_degree, clustering_coeffs)


if __name__ == "__main__":
    main()
