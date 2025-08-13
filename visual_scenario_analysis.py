import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

import sys
import subprocess

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import seaborn
    print("Seaborn is installed, version:", seaborn.__version__)
except ImportError:
    print("Seaborn not found! Attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    try:
        import seaborn
        print("Seaborn installed successfully!")
    except ImportError:
        print("Failed to install seaborn. Please check manually.")
        exit(1)


def plot_global_scenario_comparison(all_data_dict):
    """
    Aplica PCA pe toate datele combinate și colorează punctele în funcție de scenariu.
    """
    all_features = []
    all_scenario_names = []

    for scenario, features in all_data_dict.items():
        all_features.append(features)
        all_scenario_names.extend([scenario] * len(features))

    all_features = np.vstack(all_features)
    all_scenario_names = np.array(all_scenario_names)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(StandardScaler().fit_transform(all_features))

    plt.figure(figsize=(12, 8))
    unique_scenarios = np.unique(all_scenario_names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_scenarios)))
    
    for scenario, color in zip(unique_scenarios, colors):
        idx = all_scenario_names == scenario
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=scenario, alpha=0.7, s=40, color=color)

    plt.title('Comparatie PCA intre scenarii')
    plt.xlabel('Componenta principala 1')
    plt.ylabel('Componenta principala 2')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("comparatie_scenarii_pca.png", dpi=300)
    plt.show()


def plot_scenario_similarity_heatmap(all_data_dict):
    """
    Creează un heatmap al distanțelor Euclidiene între vectorii medii ai scenariilor.
    """
    scenario_names = list(all_data_dict.keys())
    num_scenarios = len(scenario_names)
    dist_matrix = np.zeros((num_scenarios, num_scenarios))

    scenario_means = [np.mean(all_data_dict[name], axis=0) for name in scenario_names]

    for i in range(num_scenarios):
        for j in range(num_scenarios):
            dist = np.linalg.norm(scenario_means[i] - scenario_means[j])
            dist_matrix[i, j] = dist

    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, xticklabels=scenario_names, yticklabels=scenario_names,
                annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Similaritate intre scenarii (distanță Euclideană între medii)")
    plt.tight_layout()
    plt.savefig("similaritate_scenarii_heatmap.png", dpi=300)
    plt.show()


def cluster_scenarios_by_mean_vectors(all_data_dict):
    """
    Aplica clustering pe vectorii medii ai fiecărui scenariu și afișează grupările.
    """
    scenario_names = list(all_data_dict.keys())
    scenario_means = [np.mean(all_data_dict[name], axis=0) for name in scenario_names]

    scaler = StandardScaler()
    scaled_means = scaler.fit_transform(scenario_means)

    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = clustering.fit_predict(scaled_means)

    print("\nGrupare scenarii pe baza vectorilor medii:")
    for scenario, label in zip(scenario_names, labels):
        print(f" - Scenariul '{scenario}' este în clusterul {label}")
