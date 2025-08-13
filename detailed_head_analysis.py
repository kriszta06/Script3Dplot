import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from disability_config import DISABILITY_MESSAGES

#sageti pentru a arata directia in care se uita subiectul
#se creeaza o sageata 3D cu coordonatele de start si end
#se converteste in 2D
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (xs[0], ys[0]), (xs[1], ys[1]), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

#convertire date de tip string din fisierele json in vectori numpy
def parse_vector_string(vector_str):
    numbers = re.findall(r'-?\d+\.?\d*', vector_str)
    if len(numbers) >= 3:
        return np.array([float(numbers[0]), float(numbers[1]), float(numbers[2])])
    return np.array([0.0, 0.0, 0.0])

def load_complete_head_data(file_path):
    #citire fisiere
    with open(file_path, 'r') as f:
        data = json.load(f)
    #declarari
    recordings = data.get('Recordings', [])
    positions = []
    rotations = []
    forward_vectors = []
    timestamps = []
    for recording in recordings:
        #extragere pozitii cap
        head_pos_str = recording.get('HeadPosition', '(0.00, 0.00, 0.00)')
        position = parse_vector_string(head_pos_str)
        positions.append(position)
        #extragere rotatii cap
        head_rot_str = recording.get('HeadRotation', '(0.00, 0.00, 0.00)')
        rotation = parse_vector_string(head_rot_str)
        rotations.append(rotation)
        #extragere vectori de directie cap
        head_forward_str = recording.get('HeadForward', '(0.00, 0.00, 1.00)')
        forward = parse_vector_string(head_forward_str)
        forward_vectors.append(forward)
        #extragere timestamp
        scene_time = recording.get('SceneTime', 0.0)
        timestamps.append(scene_time)
    #convertire liste in numpy arrays
    positions = np.array(positions)
    rotations = np.array(rotations)
    forward_vectors = np.array(forward_vectors)
    timestamps = np.array(timestamps)
    if len(timestamps) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0
    if total_duration <= 20:
        return np.array([]), np.array([]), np.array([]), np.array([])
    #ignorare primele si ultimele 10 secunde
    start_time = timestamps[0] + 10.0
    end_time = timestamps[-1] - 10.0
    valid_indices = (timestamps >= start_time) & (timestamps <= end_time)
    return positions[valid_indices], rotations[valid_indices], forward_vectors[valid_indices], timestamps[valid_indices]

#clustering hierarhic pentru analiza detaliata a capului si dendograme
def perform_agglomerative_clustering(data, scenario_name, features_combination):
    if len(data) == 0:
        print(f"No valid data for {scenario_name} - {features_combination}")
        return None
    
    print(f"\n=== CLUSTERING ANALYSIS for {scenario_name} ===")
    print(f"Features analyzed: {features_combination}")
    print(f"Total number of points: {len(data)}")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Dendrograme
    Z = linkage(scaled_data, 'ward')
    plt.figure(figsize=(15, 8))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title(f'Dendrogram for {scenario_name}\n(Shows how similar points are grouped)', fontsize=14, fontweight='bold')
    plt.xlabel('Point index', fontsize=12)
    plt.ylabel('Distance between groups', fontsize=12)
    
    # Explicatii dendograme
    plt.figtext(0.02, 0.02, 
                "DENDROGRAM EXPLANATION:\n"
                "• Each horizontal line = a group of similar points\n"
                "• Lower lines = more similar points\n"
                "• Vertical lines show how groups combine\n"
                "• Large groups (long lines) = big differences between areas",
                fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.8))
    
    plt.savefig(f'dendrogram_{scenario_name}_{features_combination.replace("+", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clustere
    n_clusters = 3
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(scaled_data)
    
    # Analiza clustere
    print(f"\nCLUSTERING RESULTS:")
    print(f"Number of groups created: {n_clusters}")
    
    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        percentage = (cluster_size / len(cluster_labels)) * 100
        print(f"  Group {i}: {cluster_size} points ({percentage:.1f}%)")
    
    if 'Position' in features_combination:
        print(f"\nCLUSTER INTERPRETATION (head positions):")
        positions = data[:, :3] if data.shape[1] >= 3 else data
        for i in range(n_clusters):
            cluster_positions = positions[cluster_labels == i]
            if len(cluster_positions) > 0:
                center = np.mean(cluster_positions, axis=0)
                print(f"  Group {i}: Center at position {center}")
                if len(cluster_positions) > 1:
                    spread = np.std(cluster_positions, axis=0)
                    print(f"           Spread: {spread}")
    
    return cluster_labels

#creeaza un plot 3D pentru clusterele detectate
def plot_clusters(positions, labels, scenario_name, combo_name):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(unique_labels)]
    
    cluster_stats = []
    
    for label, color in zip(unique_labels, colors):
        idx = labels == label
        cluster_positions = positions[idx]
        
        center = np.mean(cluster_positions, axis=0)
        size = len(cluster_positions)
        percentage = (size / len(positions)) * 100
        
        cluster_stats.append({
            'label': label,
            'color': color,
            'center': center,
            'size': size,
            'percentage': percentage
        })
        
        ax.scatter(
            cluster_positions[:, 0], cluster_positions[:, 2], cluster_positions[:, 1],
            c=[color], label=f'Area {label} ({size} points, {percentage:.1f}%)', 
            s=50, alpha=0.7
        )
        
        #Marcaj central pt clustere
        ax.scatter(center[0], center[2], center[1], 
                   c='black', s=200, marker='x', linewidth=3, zorder=10)
    
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_title(f'Head Position Grouping - {scenario_name}\n(Areas where user spent most time)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.5)
    ax.view_init(elev=30, azim=120)
    
    explanation = (
        "VISUALIZATION EXPLANATION:\n"
        "• Each color = an area where user was active\n"
        "• Large black points (X) = center of each area\n"
        "• More points in an area = more time spent there\n"
        "• Areas with many points = main areas of interest\n"
        "• Areas with few points = passing through or uninteresting areas"
    )
    
    ax.text2D(0.02, 0.98, explanation,
              transform=ax.transAxes, fontsize=10,
              va='top', bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
    
    plt.savefig(f'clusters_{scenario_name}_{combo_name.replace("+", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #Afisare statistici pentru fiecare zona de interes
    print(f"\nINTEREST AREA STATISTICS for {scenario_name}:")
    for stat in cluster_stats:
        print(f"  Area {stat['label']} ({stat['color']}):")
        print(f"    - {stat['size']} points ({stat['percentage']:.1f}% of total)")
        print(f"    - Center: X={stat['center'][0]:.2f}, Y={stat['center'][1]:.2f}, Z={stat['center'][2]:.2f}")
        if stat['percentage'] > 30:
            print(f"    - ⭐ MAIN AREA (user spent a lot of time here)")
        elif stat['percentage'] < 10:
            print(f"    - ⚠️ MINOR AREA (user spent little time here)")
        else:
            print(f"    - 📍 MODERATE AREA (medium activity)")
        print()
    
    return cluster_stats

def create_detailed_head_analysis(file_path, save_plot=True, save_csv=False, show_plot=False, global_origin=None):
    positions, rotations, forward_vectors, timestamps = load_complete_head_data(file_path)
    if len(positions) == 0:
        print(f"No valid data found in {file_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {Path(file_path).stem}")
    print(f"{'='*60}")
    
    #Normalizare a pozițiilor
    if global_origin is not None and positions.shape[1] == global_origin.shape[0]:
        positions = positions - global_origin
    else:
        origin = positions[0]
        positions = positions - origin

    N = 10
    positions = positions[::N]
    rotations = rotations[::N]
    forward_vectors = forward_vectors[::N]
    timestamps = timestamps[::N]

    print(f"📊 GENERAL STATISTICS:")
    print(f"   • Total duration: {timestamps[-1] - timestamps[0]:.1f} seconds")
    print(f"   • Number of points analyzed: {len(positions)}")
    print(f"   • Total distance traveled: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f} units")
    
    #Analiza pozitiilor
    movements = np.diff(positions, axis=0)
    movement_speeds = np.linalg.norm(movements, axis=1)
    print(f"   • Average movement speed: {np.mean(movement_speeds):.3f} units/step")
    print(f"   • Maximum speed: {np.max(movement_speeds):.3f} units/step")
    
    #Detectare viraje bruște
    turn_indices = []
    for i in range(2, len(positions)):
        v1 = positions[i-1] - positions[i-2]
        v2 = positions[i] - positions[i-1]
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            if np.degrees(angle) > 30:
                turn_indices.append(i-1)
    print(f"   • Number of sharp turns (>30°): {len(turn_indices)}")
    
    scenario_name = Path(file_path).stem
    feature_combinations = {
        "HeadPosition": positions,
        "HeadRotation": rotations,
        "HeadForward": forward_vectors,
        "SceneTimes": timestamps.reshape(-1, 1),
        "HeadPosition+HeadRotation": np.hstack((positions, rotations)),
        "HeadPosition+HeadForward": np.hstack((positions, forward_vectors)),
        "HeadRotation+HeadForward": np.hstack((rotations, forward_vectors)),
        "HeadPosition+HeadRotation+HeadForward": np.hstack((positions, rotations, forward_vectors)),
        "HeadPosition+HeadRotation+HeadForward+SceneTimes": np.hstack((positions, rotations, forward_vectors, timestamps.reshape(-1, 1)))
    }
    
    print(f"\n🔍 CLUSTERING ANALYSIS:")
    print(f"   Will analyze {len(feature_combinations)} feature combinations...")
    
    cluster_stats_list = []
    for combo_name, combo_data in feature_combinations.items():
        cluster_labels = perform_agglomerative_clustering(combo_data, scenario_name, combo_name)
        if cluster_labels is not None:
            np.savez(
                f'clustering_results_{scenario_name}_{combo_name.replace("+", "_")}.npz',
                labels=cluster_labels,
                features=combo_data
            )
            if 'Position' in combo_name:
                stats = plot_clusters(positions, cluster_labels, scenario_name, combo_name)
                cluster_stats_list.append(stats)
    
    # 3D plot
    fig = plt.figure(figsize=(22, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = positions[:, 0]
    y = positions[:, 2]  
    z = positions[:, 1]  

    ax.plot(x, y, z,
            color='deepskyblue', linewidth=2.5, alpha=0.95, linestyle='-', label='Head route')
    ax.scatter(x[0], y[0], z[0],
               c='lime', s=220, label='START', marker='o', edgecolors='black', linewidth=3, zorder=10)
    ax.scatter(x[-1], y[-1], z[-1],
               c='red', s=220, label='END', marker='*', edgecolors='black', linewidth=3, zorder=10)

    #Sageti
    step = max(1, len(positions) // 25)
    arrow_count = 0
    for i in range(0, len(positions), step):
        pos = positions[i]
        forward = forward_vectors[i]
        arrow_start = [pos[0], pos[2], pos[1]]
        arrow_end = [pos[0] + forward[0]*0.4, pos[2] + forward[2]*0.4, pos[1] + forward[1]*0.4]
        # Transmitere directă a coordonatelor
        arrow = Arrow3D(
            (arrow_start[0], arrow_end[0]),
            (arrow_start[1], arrow_end[1]),
            (arrow_start[2], arrow_end[2]), 
            mutation_scale=22, lw=2.5, arrowstyle="-|>", color="crimson", alpha=0.7)
        ax.add_artist(arrow)
        arrow_count += 1

    #Marcaje pentru unghiuri (triunghiurile galbene)
    if turn_indices:
        ax.scatter(x[turn_indices], y[turn_indices], z[turn_indices],
                   c='orange', s=80, marker='^', label=f'Turns ({len(turn_indices)} found)', zorder=9)

    ax.set_xlabel('X Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Z Position', fontsize=14, fontweight='bold')
    ax.set_zlabel('Y Position', fontsize=14, fontweight='bold')
    ax.set_title(f'3D Head Movement Route - {scenario_name}', fontsize=18, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)
    ax.view_init(elev=30, azim=120)
    
    route_explanation = (
        "3D ROUTE EXPLANATION:\n"
        "• Blue line = complete head movement path\n"
        "• Red arrows = where user was looking\n"
        "• Orange triangles = sharp turns (>30°)\n"
        "• Green point = start of experience\n"
        "• Red point = end of experience\n"
        f"• Number of arrows shown: {arrow_count}"
    )
    
    ax.text2D(0.02, 0.98, route_explanation,
              transform=ax.transAxes, fontsize=11,
              va='top', bbox=dict(facecolor='lightblue', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()

    if save_plot:
        output_file = f"detailed_analysis_{scenario_name}_3d.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n💾 3D route saved as: {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"\n✅ Analysis for {scenario_name} complete!")
    print(f"{'='*60}\n")
    
    #statistici generale
    return {
        'scenario_name': scenario_name,
        'positions': positions,
        'rotations': rotations,
        'forward_vectors': forward_vectors,
        'timestamps': timestamps,
        'total_distance': np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
        'avg_speed': np.mean(movement_speeds),
        'max_speed': np.max(movement_speeds),
        'num_sharp_turns': len(turn_indices),
        'cluster_stats': cluster_stats_list
    }

#comparație între scenarii 
def plot_all_scenarios_comparison(all_data, global_origin=None):
    if not all_data:
        print("No data available for plotting.")
        return

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = [plt.cm.tab10(i) for i in range(len(all_data))]

    for idx, (scenario_name, data) in enumerate(all_data.items()):
        # Extrage pozitiile din primele 3 coloane
        positions = data[:, :3]

        # Normalizare
        if global_origin is not None:
            positions = positions - global_origin
        else:
            positions = positions - positions[0]

        x = positions[:, 0]
        y = positions[:, 2] 
        z = positions[:, 1] 

        ax.plot(x, y, z,
                label=scenario_name,
                linewidth=2.0,
                alpha=0.85,
                color=colors[idx])

        #start
        ax.scatter(x[0], y[0], z[0], marker='o', color=colors[idx], s=100, edgecolors='black')

    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Z Position", fontsize=12)
    ax.set_zlabel("Y Position", fontsize=12)
    ax.set_title("Comparison of all scenarios 3D (head routes)", fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.view_init(elev=30, azim=120)
    plt.tight_layout()

#extragere caracteristici din datele de cap pentru detectarea dizabilităților
def extract_behavior_features(scenario_data):
    positions = scenario_data[:, :3]
    rotations = scenario_data[:, 3:6]
    forwards = scenario_data[:, 6:9]
    timestamps = scenario_data[:, -1]
    
    features = []
    
    # 1. STATISTICI SPAȚIALE DE BAZĂ
    features.extend(np.mean(positions, axis=0))  # Poziția medie
    features.extend(np.std(positions, axis=0))   # Variația poziției
    features.append(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=0)))  # Distanța totală
    
    # 2. STATISTICI DE ORIENTARE
    features.extend(np.mean(rotations, axis=0))  # Rotația medie
    features.extend(np.std(rotations, axis=0))   # Variația rotației
    features.extend(np.mean(forwards, axis=0))   # Direcția medie
    features.extend(np.std(forwards, axis=0))    # Variația direcției
    
    # 3. STATISTICI TEMPORALE
    total_time = np.max(timestamps) - np.min(timestamps)
    features.append(total_time)  # Durata totală
    features.append(np.mean(np.diff(timestamps)))  # Timpul mediu între frame-uri
    
    # 4. ANALIZA MIȘCĂRII - IMPORTANTĂ PENTRU DIZABILITĂȚI
    movements = np.diff(positions, axis=0)
    movement_speeds = np.linalg.norm(movements, axis=1)
    
    # Viteza medie și variația vitezei
    features.append(np.mean(movement_speeds))
    features.append(np.std(movement_speeds))
    features.append(np.max(movement_speeds))
    
    # Accelerarea (schimbarea vitezei)
    if len(movement_speeds) > 1:
        acceleration = np.diff(movement_speeds)
        features.append(np.mean(np.abs(acceleration)))  # Accelerarea medie
        features.append(np.std(acceleration))           # Variația accelerației
    else:
        features.extend([0, 0])
    
    # 5. ENTROPIA MIȘCĂRILOR - MĂSURĂ A COMPLEXITĂȚII
    if len(movements) > 0:
        hist, _ = np.histogramdd(movements, bins=5)
        hist = hist.flatten()
        total = np.sum(hist)
        if total > 0:
            hist = hist / total
            movement_entropy = -np.sum(hist * np.log(hist + 1e-10))
        else:
            movement_entropy = 0
    else:
        movement_entropy = 0
    features.append(movement_entropy)
    
    # 6. RAPORT DISTANȚĂ/TIMP - EFICIENȚA MIȘCĂRII
    total_distance = np.sum(movement_speeds)
    features.append(total_distance / total_time if total_time > 0 else 0)
    
    # 7. ANALIZA UNGHIURILOR - IMPORTANTĂ PENTRU COORDONARE
    pitch = rotations[:, 0]
    yaw = rotations[:, 1]
    roll = rotations[:, 2]
    
    # Variația unghiurilor
    features.append(np.ptp(pitch))  # Range-ul pitch
    features.append(np.ptp(yaw))    # Range-ul yaw
    features.append(np.ptp(roll))   # Range-ul roll
    
    # Stabilitatea unghiurilor (variația mică = mai stabil)
    features.append(np.std(pitch))
    features.append(np.std(yaw))
    features.append(np.std(roll))
    
    # 8. DETECTAREA VIRAJELOR BRUȘTE - INDICATOR DE DIZABILITATE
    sharp_turns = 0
    if len(positions) > 2:
        for i in range(2, len(positions)):
            v1 = positions[i-1] - positions[i-2]
            v2 = positions[i] - positions[i-1]
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
                if np.degrees(angle) > 30:  # Viraj > 30 grade
                    sharp_turns += 1
    features.append(sharp_turns)
    
    # 9. CONSISTENȚA MIȘCĂRII - PATTERN-URI REGULATE VS. IRREGULATE
    if len(movement_speeds) > 1:
        # Coeficientul de variație (CV) - măsoară consistența
        cv = np.std(movement_speeds) / np.mean(movement_speeds) if np.mean(movement_speeds) > 0 else 0
        features.append(cv)
        
        # Autocorelarea - măsoară pattern-urile repetitive
        if len(movement_speeds) > 10:
            autocorr = np.corrcoef(movement_speeds[:-1], movement_speeds[1:])[0, 1]
            features.append(autocorr if not np.isnan(autocorr) else 0)
        else:
            features.append(0)
    else:
        features.extend([0, 0])
    
    # 10. EFFICIENCY METRICS - CÂT DE EFICIENT ESTE COMPORTAMENTUL
    # Raportul între distanța reală și distanța euclidiană
    if len(positions) > 1:
        direct_distance = np.linalg.norm(positions[-1] - positions[0])
        actual_distance = total_distance
        efficiency = direct_distance / actual_distance if actual_distance > 0 else 0
        features.append(efficiency)
    else:
        features.append(0)
    
    # 11. STABILITATEA POZIȚIEI - CÂT DE MULT SE MIȘCĂ CAPUL
    # Variația poziției în timp
    position_variance = np.var(positions, axis=0)
    features.extend(position_variance)
    
    # 12. COMPLEXITATEA TRAIECTORIEI
    # Numărul de direcții diferite
    if len(movements) > 0:
        directions = movements / (np.linalg.norm(movements, axis=1, keepdims=True) + 1e-10)
        unique_directions = np.unique(np.round(directions, 2), axis=0)
        features.append(len(unique_directions))
    else:
        features.append(0)
    
    return np.array(features)

def detect_disability_patterns_unsupervised(all_scenarios):
    # Extrage caracteristici pentru fiecare scenariu
    feature_vectors = []
    scenario_names = []
    
    for name, data in all_scenarios.items():
        features = extract_behavior_features(data)
        feature_vectors.append(features)
        scenario_names.append(name)
    
    feature_matrix = np.array(feature_vectors)
    
    # Normalizează caracteristicile
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # Reducere dimensionalitate cu PCA
    pca = PCA(n_components=0.95)  # Păstrează 95% din varianță
    principal_components = pca.fit_transform(scaled_features)
    
    # Clusterizare DBSCAN
    clustering = DBSCAN(eps=1.5, min_samples=2).fit(principal_components)
    labels = clustering.labels_
    
    # Calculează distanțe față de centroidul principal
    centroid = np.mean(principal_components, axis=0)
    distances = cdist(principal_components, [centroid]).flatten()
    
    # Vizualizează spațiul PCA
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:
            mask = labels == label
            plt.scatter(principal_components[mask, 0], principal_components[mask, 1], 
                        c='gray', s=100, alpha=0.6, label='Outliers')
        else:
            mask = labels == label
            plt.scatter(principal_components[mask, 0], principal_components[mask, 1], 
                        s=150, alpha=0.7, label=f'Group {label}')
    
    # Adaugă numele scenariilor și distanțele
    for i, name in enumerate(scenario_names):
        plt.annotate(f"{name}\n({distances[i]:.2f})", 
                    (principal_components[i, 0], principal_components[i, 1]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
    
    plt.title('Behavioral Pattern Space (PCA)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig('disability_pattern_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculează scor de dizabilitate
    disability_likelihood = {}
    max_distance = np.max(distances)
    
    for i, name in enumerate(scenario_names):
        disability_score = float(distances[i] / max_distance)  # Convertim la float Python
        status = 'HIGH' if disability_score > 0.7 else 'MEDIUM' if disability_score > 0.4 else 'LOW'
        
        disability_likelihood[name] = {
            'cluster': int(labels[i]),          # Convertim la int Python
            'distance': float(distances[i]),    # Convertim la float Python
            'score': disability_score,
            'status': status
        }
    return disability_likelihood

def plot_disability_annotations(all_data, disability_likelihood, global_origin=None):
    """Adaugă marker vizual în plot-ul 3D pentru dizabilități"""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Culori bazate pe scorul de dizabilitate
    colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
    
    for scenario_name, data in all_data.items():
        # Extrage pozițiile (primele 3 coloane)
        positions = data[:, :3]
        
        # Normalizează față de global_origin dacă există
        if global_origin is not None:
            positions = positions - global_origin
        else:
            positions = positions - positions[0]
        
        # Plotează traseul
        x = positions[:, 0]
        y = positions[:, 2]  
        z = positions[:, 1]  
        
        # Alege culoarea în funcție de dizabilitate
        status = disability_likelihood[scenario_name]['status']

        ax.plot(x, y, z, color=colors[status], alpha=0.7, linewidth=2.5)
        
        # Adaugă marker la început
        ax.scatter(x[0], y[0], z[0], color=colors[status], s=120, marker='o', edgecolors='black')
        
        # Adaugă etichetă cu scorul
        score = disability_likelihood[scenario_name]['score']
        ax.text(x[0], y[0], z[0], 
                f"{scenario_name}: {score:.2f}",
                fontsize=9, zorder=10)
    
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_title('3D Head Paths - Disability Likelihood', fontsize=16, fontweight='bold')
    
    # Adaugă legendă
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='High Disability Likelihood'),
        Line2D([0], [0], color='orange', lw=2, label='Medium Disability Likelihood'),
        Line2D([0], [0], color='green', lw=2, label='Low Disability Likelihood')
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    
    # Explicație
    explanation = (
        "DISABILITY LIKELIHOOD BASED ON BEHAVIORAL PATTERNS:\n"
        "• RED: Significantly different behavior (high disability likelihood)\n"
        "• ORANGE: Moderately different behavior\n"
        "• GREEN: Typical behavior (low disability likelihood)"
    )
    ax.text2D(0.02, 0.98, explanation, 
              transform=ax.transAxes, fontsize=10,
              bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig('disability_3d_assessment.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(disability_assessment, all_person_features):
    """Salvează rezultatele detaliate în format JSON"""
    detailed_results = {
        'analysis_timestamp': str(np.datetime64('now')),
        'total_persons': len(disability_assessment),
        'methodology': {
            'pca_components': '95% variance retained',
            'clustering': 'DBSCAN with eps=1.5, min_samples=2',
            'scoring': 'Combined: 50% distance + 30% Mahalanobis + 20% consistency',
            'thresholds': {
                'severe': '>0.7',
                'moderate': '>0.5',
                'light': '>0.3',
                'none': '≤0.3'
            }
        },
        'persons': {}
    }
    
    for person_id, result in disability_assessment.items():
        detailed_results['persons'][person_id] = {
            'disability_assessment': result,
            'raw_features': all_person_features[person_id].tolist() if person_id in all_person_features else []
        }
    
    # Salvează rezultatele principale
    with open('detailed_disability_assessment.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Salvează un rezumat simplu
    summary = {
        'summary': {
            'total_persons': len(disability_assessment),
            'severe_disability': len([p for p in disability_assessment.values() if p['status'] == 'HIGH']),
            'moderate_disability': len([p for p in disability_assessment.values() if p['status'] == 'MEDIUM']),
            'light_disability': len([p for p in disability_assessment.values() if p['status'] == 'LOW']),
            'no_disability': len([p for p in disability_assessment.values() if p['status'] == 'NONE'])
        },
        'persons': {}
    }
    
    for person_id, result in disability_assessment.items():
        summary['persons'][person_id] = {
            'status': result['status'],
            'disability_level': result['disability_level'],
            'final_score': result['final_score'],
            'risk_factors': result['risk_factors']
        }
    
    with open('disability_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Rezultatele au fost salvate în:")
    print(f"   • detailed_disability_assessment.json - Analiză completă")
    print(f"   • disability_summary.json - Rezumat pentru utilizare rapidă")

def create_visual_disability_report(disability_assessment):
    """Creează un raport vizual simplificat cu doar 2 grafice"""
    try:
        # Pregătește datele pentru grafice
        persons = list(disability_assessment.keys())
        
        # Verifică dacă există date valide
        if not persons:
            print("❌ Nu există date pentru generarea raportului vizual")
            return
            
        # Extrage scorurile finale
        final_scores = []
        valid_persons = []
        
        for p in persons:
            try:
                final_score = disability_assessment[p]['final_score']
                # Verifică dacă scorurile sunt valide (nu NaN)
                if not np.isnan(final_score):
                    final_scores.append(float(final_score))
                    valid_persons.append(p)
            except (KeyError, ValueError, TypeError) as e:
                print(f"⚠️ Eroare la extragerea datelor pentru {p}: {e}")
                continue
        
        if not valid_persons:
            print("❌ Nu există date valide pentru generarea graficelelor")
            return
            
        # Status-uri pentru culori
        status_colors = {
            'HIGH': 'red',
            'MEDIUM': 'orange', 
            'LOW': 'yellow',
            'NONE': 'green'
        }
        colors = [status_colors[disability_assessment[p]['status']] for p in valid_persons]
        
        # Figura principală cu 2 subplot-uri
        fig = plt.figure(figsize=(16, 8))
        
        # 1. Scorurile finale comparate
        ax1 = plt.subplot(1, 2, 1)
        bars = ax1.bar(range(len(valid_persons)), final_scores, color=colors, alpha=0.7)
        ax1.set_title('Scorurile Finale de Dizabilitate', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Persoane')
        ax1.set_ylabel('Scor Final')
        ax1.set_xticks(range(len(valid_persons)))
        ax1.set_xticklabels([p.replace('Person_', 'P') for p in valid_persons], rotation=45)
        ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Sever (0.7)')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderat (0.5)')
        ax1.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.7, label='Ușor (0.3)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Adaugă valorile pe bare
        for i, (bar, score) in enumerate(zip(bars, final_scores)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Rankingul persoanelor în funcție de severitate
        ax2 = plt.subplot(1, 2, 2)
        sorted_indices = np.argsort(final_scores)[::-1]  # Sortare descrescătoare
        sorted_scores = [final_scores[i] for i in sorted_indices]
        sorted_persons = [valid_persons[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        
        bars = ax2.bar(range(len(sorted_persons)), sorted_scores, 
                       color=sorted_colors, alpha=0.7)
        ax2.set_title('Ranking Persoane după Severitatea Dizabilității', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Ranking')
        ax2.set_ylabel('Scor Final')
        ax2.set_xticks(range(len(sorted_persons)))
        ax2.set_xticklabels([f"{i+1}. {p.replace('Person_', 'P')}" for i, p in enumerate(sorted_persons)], 
                           rotation=45)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Sever (0.7)')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderat (0.5)')
        ax2.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.7, label='Ușor (0.3)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adaugă valorile pe bare
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Titlu general
        plt.suptitle('Raport Vizual Dizabilități - Analiză Comportamentală', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig('disability_visual_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Raportul vizual simplificat a fost salvat ca: disability_visual_report.png")
        print(f"   • {len(valid_persons)} persoane afișate")
        print(f"   • Scoruri finale: {min(final_scores):.3f} - {max(final_scores):.3f}")
        
    except Exception as e:
        print(f"❌ Eroare la generarea raportului vizual: {e}")
        import traceback
        traceback.print_exc()
       
def group_files_by_person(json_files):
    """Grupează fișierele după identificatorul de persoană"""
    grouped = {}
    for file_path in json_files:
        # Extrage identificatorul de persoană din formatul '1-A.json', '2-B.json', etc.
        match = re.search(r'(\d+)-[A-Za-z]+\.json$', file_path.name)
        if match:
            person_id = f"Person_{match.group(1)}"
            if person_id not in grouped:
                grouped[person_id] = []
            grouped[person_id].append(file_path)
    
    # Debug: afișează ce s-a găsit
    print(f"DEBUG: {len(json_files)} fișiere procesate")
    print(f"DEBUG: {len(grouped)} persoane găsite")
    for person, files in grouped.items():
        print(f"DEBUG: {person}: {len(files)} fișiere")
    
    return grouped

def combine_features_for_person(scenario_features):
    """Combină caracteristicile din cele 3 scenarii într-un singur vector pentru persoană"""
    combined = []
    # Adaugă toate caracteristicile din scenarii într-un singur vector
    for features in scenario_features.values():
        combined.extend(features)
    return np.array(combined)

def analyze_person_disability(all_person_features):
    # Extrage caracteristicile pentru fiecare persoană
    feature_vectors = []
    person_ids = []
    
    for person_id, features in all_person_features.items():
        feature_vectors.append(features)
        person_ids.append(person_id)
    
    feature_matrix = np.array(feature_vectors)
    
    # Normalizează caracteristicile
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # METODA 1: PCA + DBSCAN pentru detectarea de anomalii
    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(scaled_features)
    
    # Clusterizare DBSCAN pentru detectia de anomalii
    clustering = DBSCAN(eps=1.5, min_samples=2).fit(principal_components)
    labels = clustering.labels_
    
    # Calculează distanțe față de centroid
    centroid = np.mean(principal_components, axis=0)
    distances = cdist(principal_components, [centroid]).flatten()
    
    # METODA 2: Analiza multivariată pentru detectarea outlier-elor
    # Calculează scorul Mahalanobis pentru fiecare persoană
    try:
        # Covarianța matricei de caracteristici
        cov_matrix = np.cov(scaled_features.T)
        
        # Verifică dacă matricea de covarianță este inversabilă și are valori finite
        if np.linalg.det(cov_matrix) > 1e-10 and np.all(np.isfinite(cov_matrix)):
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            
            # Scorul Mahalanobis pentru fiecare persoană
            mahalanobis_scores = []
            for i, features in enumerate(scaled_features):
                diff = features - np.mean(scaled_features, axis=0)
                mahal_score = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                
                # Verifică dacă scorul este valid
                if np.isnan(mahal_score) or np.isinf(mahal_score):
                    mahal_score = distances[i]  # Fallback la distanța euclidiană
                
                mahalanobis_scores.append(mahal_score)
            
            mahalanobis_scores = np.array(mahalanobis_scores)
        else:
            # Dacă matricea de covarianță nu este inversabilă, folosește distanțele euclidiene
            mahalanobis_scores = distances
    except:
        # Dacă matricea de covarianță nu este inversabilă, folosește distanțele euclidiene
        mahalanobis_scores = distances
    
    # METODA 3: Analiza de consistență între scenarii
    # Calculează cât de consistent este comportamentul fiecărei persoane
    consistency_scores = []
    for i, features in enumerate(scaled_features):
        # Calculează cât de departe este de alte persoane
        other_features = np.vstack([scaled_features[j] for j in range(len(scaled_features)) if j != i])
        if len(other_features) > 0:
            # Distanța medie față de alte persoane
            avg_distance_to_others = np.mean(cdist([features], other_features))
            consistency_scores.append(avg_distance_to_others)
        else:
            consistency_scores.append(0)
    
    consistency_scores = np.array(consistency_scores)
    
    # COMBINĂ TOATE METODELE PENTRU UN SCOR FINAL
    # Normalizează toate scorurile la [0,1] cu verificări pentru a evita division by zero
    max_distance = np.max(distances) if len(distances) > 0 and np.max(distances) > 0 else 1
    max_mahal = np.max(mahalanobis_scores) if len(mahalanobis_scores) > 0 and np.max(mahalanobis_scores) > 0 else 1
    max_consistency = np.max(consistency_scores) if len(consistency_scores) > 0 and np.max(consistency_scores) > 0 else 1
    
    # Scorul final combinat (media ponderată)
    final_scores = []
    normalized_scores = []  # Stochează scorurile normalizate pentru fiecare persoană
    
    for i in range(len(person_ids)):
        # Normalizează scorurile cu verificări pentru a evita NaN
        norm_distance = distances[i] / max_distance if max_distance > 0 else 0
        norm_mahal = mahalanobis_scores[i] / max_mahal if max_mahal > 0 else 0
        norm_consistency = consistency_scores[i] / max_consistency if max_consistency > 0 else 0
        
        # Verifică dacă scorurile sunt valide (nu NaN sau inf)
        if np.isnan(norm_distance) or np.isinf(norm_distance):
            norm_distance = 0
        if np.isnan(norm_mahal) or np.isinf(norm_mahal):
            norm_mahal = 0
        if np.isnan(norm_consistency) or np.isinf(norm_consistency):
            norm_consistency = 0
        
        # Media ponderată (distanța are ponderea cea mai mare)
        final_score = 0.5 * norm_distance + 0.3 * norm_mahal + 0.2 * norm_consistency
        
        # Verifică dacă scorul final este valid
        if np.isnan(final_score) or np.isinf(final_score):
            final_score = 0.0
        
        final_scores.append(final_score)
        
        # Salvează scorurile normalizate pentru a le folosi mai târziu
        normalized_scores.append({
            'norm_distance': norm_distance,
            'norm_mahal': norm_mahal,
            'norm_consistency': norm_consistency
        })
    
    final_scores = np.array(final_scores)
    
    # Verifică și corectează orice scor final care ar putea fi NaN
    for i in range(len(final_scores)):
        if np.isnan(final_scores[i]) or np.isinf(final_scores[i]):
            final_scores[i] = 0.0
    
    # Calculează scor de dizabilitate
    disability_assessment = {}
    
    for i, person_id in enumerate(person_ids):
        # Scorul final normalizat
        disability_score = final_scores[i]
        
        # Status bazat pe scorul final
        if disability_score > 0.7:
            status = 'HIGH'
            disability_level = DISABILITY_MESSAGES['HIGH']
        elif disability_score > 0.5:
            status = 'MEDIUM'
            disability_level = DISABILITY_MESSAGES['MEDIUM']
        elif disability_score > 0.3:
            status = 'LOW'
            disability_level = DISABILITY_MESSAGES['LOW']
        else:
            status = 'NONE'
            disability_level = DISABILITY_MESSAGES['NONE']
        
        # Verifică și corectează scorurile finale pentru a evita NaN
        final_score_clean = disability_score
        if np.isnan(final_score_clean) or np.isinf(final_score_clean):
            final_score_clean = 0.0
        
        # Verifică și corectează scorurile individuale
        distance_score_clean = distances[i]
        if np.isnan(distance_score_clean) or np.isinf(distance_score_clean):
            distance_score_clean = 0.0
            
        mahal_score_clean = mahalanobis_scores[i]
        if np.isnan(mahal_score_clean) or np.isinf(mahal_score_clean):
            mahal_score_clean = 0.0
            
        consistency_score_clean = consistency_scores[i]
        if np.isnan(consistency_score_clean) or np.isinf(consistency_score_clean):
            consistency_score_clean = 0.0
        
        # Detalii suplimentare pentru analiză
        disability_assessment[person_id] = {
            'cluster': int(labels[i]),
            'distance_score': float(distance_score_clean),
            'mahalanobis_score': float(mahal_score_clean),
            'consistency_score': float(consistency_score_clean),
            'final_score': float(final_score_clean),
            'status': status,
            'disability_level': disability_level,
            'disability': status in ['HIGH', 'MEDIUM'],
            'risk_factors': []
        }
        
        # Identifică factorii de risc folosind scorurile normalizate salvate
        norm_scores = normalized_scores[i]
        if norm_scores['norm_distance'] > 0.7:
            disability_assessment[person_id]['risk_factors'].append('COMPORTAMENT ABNORMAL')
        if norm_scores['norm_mahal'] > 0.7:
            disability_assessment[person_id]['risk_factors'].append('PATTERN-URI NESTANDARD')
        if norm_scores['norm_consistency'] > 0.7:
            disability_assessment[person_id]['risk_factors'].append('INCONSISTENȚĂ ÎNTRE SCENARII')
        
        # Nu mai adăugăm recomandări
    
    return disability_assessment

def create_disability_report(disability_assessment):
    """Creează un raport detaliat al dizabilităților"""
    print("\n" + "="*100)
    print("🔍 DETALIAT DISABILITY ASSESSMENT REPORT - ANALIZĂ COMPORTAMENTALĂ")
    print("="*100)
    
    # Grupează persoanele după status
    high_disability = []
    medium_disability = []
    low_disability = []
    no_disability = []
    
    for person, result in disability_assessment.items():
        if result['status'] == 'HIGH':
            high_disability.append((person, result))
        elif result['status'] == 'MEDIUM':
            medium_disability.append((person, result))
        elif result['status'] == 'LOW':
            low_disability.append((person, result))
        else:
            no_disability.append((person, result))
    
    # Afișează persoanele cu dizabilități severe
    if high_disability:
        print(f"\n🔴 PERSOANE CU DIZABILITĂȚI SEVERE ({len(high_disability)}):")
        print("-" * 70)
        for person, result in sorted(high_disability, key=lambda x: x[1]['final_score'], reverse=True):
            print(f"  • {person}:")
            print(f"    Score final: {result['final_score']:.3f} | Cluster: {result['cluster']}")
            print(f"    Nivel: {result['disability_level']}")
            print(f"    Distanța: {result['distance_score']:.3f} | Mahalanobis: {result['mahalanobis_score']:.3f}")
            print(f"    Consistența: {result['consistency_score']:.3f}")
            if result['risk_factors']:
                print(f"    Factorii de risc: {', '.join(result['risk_factors'])}")
            print()
    
    # Afișează persoanele cu dizabilități moderate
    if medium_disability:
        print(f"\n🟠 PERSOANE CU DIZABILITĂȚI MODERATE ({len(medium_disability)}):")
        print("-" * 70)
        for person, result in sorted(medium_disability, key=lambda x: x[1]['final_score'], reverse=True):
            print(f"  • {person}:")
            print(f"    Score final: {result['final_score']:.3f} | Cluster: {result['cluster']}")
            print(f"    Nivel: {result['disability_level']}")
            print(f"    Distanța: {result['distance_score']:.3f} | Mahalanobis: {result['mahalanobis_score']:.3f}")
            print(f"    Consistența: {result['consistency_score']:.3f}")
            if result['risk_factors']:
                print(f"    Factorii de risc: {', '.join(result['risk_factors'])}")
            print()
    
    # Afișează persoanele cu dizabilități ușoare
    if low_disability:
        print(f"\n🟡 PERSOANE CU DIZABILITĂȚI UȘOARE ({len(low_disability)}):")
        print("-" * 70)
        for person, result in sorted(low_disability, key=lambda x: x[1]['final_score'], reverse=True):
            print(f"  • {person}:")
            print(f"    Score final: {result['final_score']:.3f} | Cluster: {result['cluster']}")
            print(f"    Nivel: {result['disability_level']}")
            print(f"    Distanța: {result['distance_score']:.3f} | Mahalanobis: {result['mahalanobis_score']:.3f}")
            print(f"    Consistența: {result['consistency_score']:.3f}")
            if result['risk_factors']:
                print(f"    Factorii de risc: {', '.join(result['risk_factors'])}")
            print()
    
    # Afișează persoanele fără dizabilități
    if no_disability:
        print(f"\n🟢 PERSOANE FĂRĂ DIZABILITĂȚI ({len(no_disability)}):")
        print("-" * 70)
        for person, result in sorted(no_disability, key=lambda x: x[1]['final_score']):
            print(f"  • {person}:")
            print(f"    Score final: {result['final_score']:.3f} | Cluster: {result['cluster']}")
            print(f"    Nivel: {result['disability_level']}")
            print(f"    Distanța: {result['distance_score']:.3f} | Mahalanobis: {result['mahalanobis_score']:.3f}")
            print(f"    Consistența: {result['consistency_score']:.3f}")
            print()
    
    # Statistici generale
    total_persons = len(disability_assessment)
    persons_with_disability = len(high_disability) + len(medium_disability) + len(low_disability)
    percentage_with_disability = (persons_with_disability / total_persons) * 100
    
    print(f"\n📊 STATISTICI GENERALE:")
    print("-" * 50)
    print(f"  • Total persoane analizate: {total_persons}")
    print(f"  • Persoane cu dizabilități severe: {len(high_disability)} ({len(high_disability)/total_persons*100:.1f}%)")
    print(f"  • Persoane cu dizabilități moderate: {len(medium_disability)} ({len(medium_disability)/total_persons*100:.1f}%)")
    print(f"  • Persoane cu dizabilități ușoare: {len(low_disability)} ({len(low_disability)/total_persons*100:.1f}%)")
    print(f"  • Persoane fără dizabilități: {len(no_disability)} ({len(no_disability)/total_persons*100:.1f}%)")
    print(f"  • Total persoane cu dizabilități: {persons_with_disability} ({percentage_with_disability:.1f}%)")
    
    # Analiza cluster-elor
    clusters = {}
    for person, result in disability_assessment.items():
        cluster = result['cluster']
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(person)
    
    print(f"\n🔗 ANALIZA CLUSTER-ELOR:")
    print("-" * 30)
    for cluster_id, persons in clusters.items():
        cluster_statuses = [disability_assessment[p]['status'] for p in persons]
        high_count = cluster_statuses.count('HIGH')
        medium_count = cluster_statuses.count('MEDIUM')
        low_count = cluster_statuses.count('LOW')
        none_count = cluster_statuses.count('NONE')
        
        print(f"  • Cluster {cluster_id}: {len(persons)} persoane")
        print(f"    - Severe: {high_count}, Moderate: {medium_count}, Ușoare: {low_count}, Fără: {none_count}")
    
    # Recomandări generale
    print(f"\n💡 RECOMANDĂRI GENERALE:")
    print("-" * 30)
    if high_disability:
        print(f"  • URGENT: {len(high_disability)} persoane necesită evaluare medicală imediată")
    if medium_disability:
        print(f"  • IMPORTANT: {len(medium_disability)} persoane necesită monitorizare și evaluare")
    if low_disability:
        print(f"  • ATENȚIE: {len(low_disability)} persoane necesită monitorizare ușoară")
    if no_disability:
        print(f"  • NORMAL: {len(no_disability)} persoane au comportament tipic")
    
    # Explicație metodologie
    print(f"\n🔬 METODOLOGIA ANALIZEI:")
    print("-" * 30)
    print(f"  • Analiză PCA + DBSCAN pentru detectarea de anomalii")
    print(f"  • Scorul Mahalanobis pentru pattern-uri nestandard")
    print(f"  • Analiza de consistență între scenarii A, B, C")
    print(f"  • Scorul final combinat (50% distanță + 30% Mahalanobis + 20% consistență)")
    print(f"  • Threshold-uri: >0.7 (sever), >0.5 (moderat), >0.3 (ușor), ≤0.3 (fără)")
    
    print("="*100)

def cleanup_disability_files():
    """Șterge doar fișierele legate de analiza dizabilităților și comparațiile de scenarii"""
    import os
    import glob
    
    # Lista fișierelor legate de dizabilități care trebuie șterse
    disability_files = [
        'disability_visual_report.png',
        'detailed_disability_assessment.json',
        'disability_summary.json'
    ]
    
    # Lista fișierelor de comparație scenarii care trebuie regenerate
    scenario_comparison_files = [
        'Person_*_scenarios_comparison.png'
    ]
    
    print("🧹 Curățare fișiere legate de dizabilități și comparații scenarii...")
    
    # Șterge fișierele de dizabilități
    for file_pattern in disability_files:
        # Caută fișierele care se potrivesc pattern-ului
        matching_files = glob.glob(file_pattern)
        for file_path in matching_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"   ✅ Șters: {file_path}")
            except Exception as e:
                print(f"   ❌ Eroare la ștergerea {file_path}: {e}")
    
    # Șterge fișierele de comparație scenarii
    for file_pattern in scenario_comparison_files:
        # Caută fișierele care se potrivesc pattern-ului
        matching_files = glob.glob(file_pattern)
        for file_path in matching_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"   ✅ Șters: {file_path}")
            except Exception as e:
                print(f"   ❌ Eroare la ștergerea {file_path}: {e}")
    
    print("   🎯 Fișierele din prima variantă au fost păstrate!")
    print("   🔄 Fișierele de comparație scenarii vor fi regenerate!")

def main():
    # Curăță doar fișierele legate de dizabilități și comparații scenarii
    cleanup_disability_files()
    
    json_files = list(Path('vr_recordings').glob('*.json'))
    if not json_files:
        print("No JSON files found in vr_recordings directory")
        return

    # Grupează fișierele după persoană
    person_files = group_files_by_person(json_files)
    
    if not person_files:
        print("No valid person files found")
        return
    
    print(f"Found {len(person_files)} persons with scenarios")
    
    # Debug: afișează ce fișiere au fost găsite pentru fiecare persoană
    for person_id, files in person_files.items():
        print(f"  • {person_id}: {[f.name for f in files]}")
    
    all_person_features = {}
    disability_assessment = {}
    
    # Procesează fiecare persoană
    for person_id, files in person_files.items():
        print(f"\n{'='*40}")
        print(f"PROCESSING PERSON: {person_id}")
        print(f"{'='*40}")
        
        person_features = {}
        all_data = {}
        global_origin = None
        
        # Procesează fiecare scenariu al persoanei
        for file_path in files:
            scenario_name = file_path.stem
            print(f"\nProcessing scenario: {scenario_name}")
            
            try:
                # Încarcă datele
                positions, rotations, forward_vectors, timestamps = load_complete_head_data(str(file_path))

                if len(positions) == 0:
                    print(f"No valid data for scenario {scenario_name}")
                    continue

                # Setează originea globală dacă nu există
                if global_origin is None:
                    global_origin = positions[0]

                # Normalizează pozițiile
                positions = positions - global_origin

                # Sub-eșantionează
                N = 10
                positions = positions[::N]
                rotations = rotations[::N]
                forward_vectors = forward_vectors[::N]
                timestamps = timestamps[::N]

                # Salvează datele pentru analiza globală
                scenario_data = np.hstack((
                    positions,
                    rotations,
                    forward_vectors,
                    timestamps.reshape(-1, 1)
                ))
                all_data[scenario_name] = scenario_data

                # Extrage caracteristicile pentru scenariu
                features = extract_behavior_features(scenario_data)
                print(f"      ✅ Caracteristici extrase: {len(features)} features")
                person_features[scenario_name] = features
                
                # Verifică dacă fișierele de analiză individuală există deja
                individual_analysis_file = f"detailed_analysis_{scenario_name}_3d.png"
                if not os.path.exists(individual_analysis_file):
                    print(f"   🔄 Generare analiză individuală pentru {scenario_name}...")
                    create_detailed_head_analysis(file_path, save_plot=True, global_origin=global_origin, show_plot=False)
                else:
                    print(f"   ✅ Analiza individuală pentru {scenario_name} există deja")

            except Exception as e:
                print(f"Error processing {scenario_name}: {e}")
        
        # Combină caracteristicile din cele 3 scenarii
        if person_features:
            print(f"   🔄 Combinare caracteristici pentru {person_id}...")
            combined_features = combine_features_for_person(person_features)
            print(f"      ✅ Caracteristici combinate: {len(combined_features)} features")
            all_person_features[person_id] = combined_features
            
            # Plot comparativ pentru scenariile aceleiași persoane
            if all_data:
                scenario_comparison_file = f"{person_id}_scenarios_comparison.png"
                if not os.path.exists(scenario_comparison_file):
                    print(f"   🔄 Generare comparație scenarii pentru {person_id}...")
                    plot_all_scenarios_comparison(all_data, global_origin=global_origin)
                    plt.savefig(scenario_comparison_file, dpi=300)
                    plt.close()
                else:
                    print(f"   ✅ Comparația scenarii pentru {person_id} există deja")
    
    # Analizează dizabilitățile pe baza caracteristicilor combinate
    if all_person_features:
        print(f"\n🔍 ANALIZA DIZABILITĂȚILOR...")
        print(f"   • Caracteristici disponibile pentru {len(all_person_features)} persoane")
        
        # Debug: afișează ce persoane au caracteristici
        for person_id, features in all_person_features.items():
            print(f"      - {person_id}: {len(features)} caracteristici")
        
        disability_assessment = analyze_person_disability(all_person_features)
        print(f"   • Analiza dizabilităților finalizată pentru {len(disability_assessment)} persoane")
        
        # Salvează rezultatele
        print(f"   💾 Salvare rezultate...")
        save_detailed_results(disability_assessment, all_person_features)
        
        print(f"   📊 Generare raport vizual...")
        create_visual_disability_report(disability_assessment)
        
        # Afișează rezultatele finale
        print(f"   📋 Afișare raport final...")
        create_disability_report(disability_assessment)
        
        print(f"\n🎉 ANALIZA COMPLETĂ FINALIZATĂ!")
        print(f"   • {len(person_files)} persoane procesate")
        print(f"   • Fișierele din prima variantă au fost păstrate")
        print(f"   • Fișierele de comparație scenarii au fost regenerate")
        print(f"   • Imaginile de dizabilități au fost regenerate cu graficele corecte")
    else:
        print(f"\n❌ NU SUNT CARACTERISTICI DISPONIBILE PENTRU ANALIZA DIZABILITĂȚILOR!")
        print(f"   • Verifică dacă datele au fost procesate corect")
        print(f"   • Verifică dacă fișierele JSON conțin date valide")

if __name__ == "__main__":
    main()