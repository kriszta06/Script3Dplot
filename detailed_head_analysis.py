import sys
from pathlib import Path
import json
import numpy as np
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

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        # Corecție: înlocuiți time(xs[1], ys[1]) cu (xs[1], ys[1])
        FancyArrowPatch.__init__(self, (xs[0], ys[0]), (xs[1], ys[1]), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def parse_vector_string(vector_str):
    numbers = re.findall(r'-?\d+\.?\d*', vector_str)
    if len(numbers) >= 3:
        return np.array([float(numbers[0]), float(numbers[1]), float(numbers[2])])
    return np.array([0.0, 0.0, 0.0])

def load_complete_head_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    recordings = data.get('Recordings', [])
    positions = []
    rotations = []
    forward_vectors = []
    timestamps = []
    for recording in recordings:
        head_pos_str = recording.get('HeadPosition', '(0.00, 0.00, 0.00)')
        position = parse_vector_string(head_pos_str)
        positions.append(position)
        head_rot_str = recording.get('HeadRotation', '(0.00, 0.00, 0.00)')
        rotation = parse_vector_string(head_rot_str)
        rotations.append(rotation)
        head_forward_str = recording.get('HeadForward', '(0.00, 0.00, 1.00)')
        forward = parse_vector_string(head_forward_str)
        forward_vectors.append(forward)
        scene_time = recording.get('SceneTime', 0.0)
        timestamps.append(scene_time)
    positions = np.array(positions)
    rotations = np.array(rotations)
    forward_vectors = np.array(forward_vectors)
    timestamps = np.array(timestamps)
    if len(timestamps) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Corecție: mutați calculul duratei totale aici
    total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0
    if total_duration <= 20:
        return np.array([]), np.array([]), np.array([]), np.array([])
    start_time = timestamps[0] + 10.0
    end_time = timestamps[-1] - 10.0
    valid_indices = (timestamps >= start_time) & (timestamps <= end_time)
    return positions[valid_indices], rotations[valid_indices], forward_vectors[valid_indices], timestamps[valid_indices]

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
    
    # Return statistics for disability assessment
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

        # Corecție: înlocuiți colors(idx) cu colors[idx]
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
    plt.savefig("all_scenarios_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved combined 3D plot: all_scenarios_comparison.png")
    plt.close()

# =============================================================================
# DISABILITY DETECTION FUNCTIONS (NON-SUPERVISED WITHOUT KERAS)
# =============================================================================

def extract_behavior_features(scenario_data):
    """Extrage caracteristici temporale și spațiale din date"""
    positions = scenario_data[:, :3]
    rotations = scenario_data[:, 3:6]
    forwards = scenario_data[:, 6:9]
    timestamps = scenario_data[:, -1]
    
    features = []
    
    # Statistici spațiale
    features.extend(np.mean(positions, axis=0))
    features.extend(np.std(positions, axis=0))
    features.append(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=0)))
    
    # Statistici de orientare
    features.extend(np.mean(rotations, axis=0))
    features.extend(np.std(forwards, axis=0))
    
    # Statistici temporale
    features.append(np.max(timestamps) - np.min(timestamps))
    features.append(np.mean(np.diff(timestamps)))
    
    # Entropia mișcărilor
    movement = np.diff(positions, axis=0)
    hist, _ = np.histogramdd(movement, bins=5)
    hist = hist.flatten()
    # Protecție împărțire la zero
    total = np.sum(hist)
    if total > 0:
        hist = hist / total
    else:
        hist = np.zeros_like(hist)
    movement_entropy = -np.sum(hist * np.log(hist + 1e-10))
    features.append(movement_entropy)
    
    # Raport distanță/timp
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=0))
    total_time = timestamps[-1] - timestamps[0]
    features.append(total_distance / total_time if total_time > 0 else 0)
    
    # Variația unghiului
    pitch = rotations[:, 0]
    yaw = rotations[:, 1]
    features.append(np.ptp(pitch))
    features.append(np.ptp(yaw))
    
    return np.array(features)

def detect_disability_patterns_unsupervised(all_scenarios):
    """Detectă pattern-uri de dizabilitate folosind PCA și DBSCAN"""
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
        y = positions[:, 2]   # Z este pe axa Y în plot
        z = positions[:, 1]   # Y este pe axa Z în plot
        
        # Alege culoarea în funcție de dizabilitate
        status = disability_likelihood[scenario_name]['status']
        # Corecție: înlocuiți colors[i] cu colors[status]
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

# ... (codul existent până la funcția main) ...

def group_files_by_person(json_files):
    """Grupează fișierele după identificatorul de persoană"""
    grouped = {}
    for file_path in json_files:
        # Extrage identificatorul de persoană (ex: 'person1' din 'person1_A.json')
        match = re.search(r'(.+?)_[A-Za-z]+\.json$', file_path.stem)
        if match:
            person_id = match.group(1)
            if person_id not in grouped:
                grouped[person_id] = []
            grouped[person_id].append(file_path)
    return grouped

def combine_features_for_person(scenario_features):
    """Combină caracteristicile din cele 3 scenarii într-un singur vector pentru persoană"""
    combined = []
    # Adaugă toate caracteristicile din scenarii într-un singur vector
    for features in scenario_features.values():
        combined.extend(features)
    return np.array(combined)

def analyze_person_disability(all_person_features):
    """Analizează dizabilitățile pe baza caracteristicilor combinate ale fiecărei persoane"""
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
    
    # Reducere dimensionalitate cu PCA
    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(scaled_features)
    
    # Clusterizare DBSCAN pentru detectia de anomalii
    clustering = DBSCAN(eps=1.5, min_samples=2).fit(principal_components)
    labels = clustering.labels_
    
    # Calculează distanțe față de centroid
    centroid = np.mean(principal_components, axis=0)
    distances = cdist(principal_components, [centroid]).flatten()
    
    # Calculează scor de dizabilitate
    disability_assessment = {}
    max_distance = np.max(distances) if len(distances) > 0 else 1
    
    for i, person_id in enumerate(person_ids):
        disability_score = distances[i] / max_distance
        status = 'HIGH' if disability_score > 0.7 else 'MEDIUM' if disability_score > 0.4 else 'LOW'
        
        disability_assessment[person_id] = {
            'cluster': int(labels[i]),
            'distance': float(distances[i]),
            'score': disability_score,
            'status': status,
            'disability': status in ['HIGH', 'MEDIUM']
        }
    
    return disability_assessment

def main():
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
                person_features[scenario_name] = features
                
                # Analiză detaliată individuală (opțional)
                # create_detailed_head_analysis(file_path, save_plot=True, global_origin=global_origin, show_plot=False)

            except Exception as e:
                print(f"Error processing {scenario_name}: {e}")
        
        # Combină caracteristicile din cele 3 scenarii
        if person_features:
            combined_features = combine_features_for_person(person_features)
            all_person_features[person_id] = combined_features
            
            # Plot comparativ pentru scenariile aceleiași persoane
            if all_data:
                plot_all_scenarios_comparison(all_data, global_origin=global_origin)
                plt.savefig(f"{person_id}_scenarios_comparison.png", dpi=300)
                plt.close()
    
    # Analizează dizabilitățile pe baza caracteristicilor combinate
    if all_person_features:
        disability_assessment = analyze_person_disability(all_person_features)
        
        # Salvează rezultatele
        with open('person_disability_assessment.json', 'w') as f:
            json.dump(disability_assessment, f, indent=2)
        
        # Afișează rezultatele finale
        print("\n\n" + "="*60)
        print("PERSON DISABILITY ASSESSMENT RESULTS")
        print("="*60)
        for person, result in disability_assessment.items():
            disability_status = "WITH DISABILITY" if result['disability'] else "NO DISABILITY"
            print(f"{person}: {disability_status} (score: {result['score']:.2f}, status: {result['status']})")

if __name__ == "__main__":
    main()