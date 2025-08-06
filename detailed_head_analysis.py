import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from pathlib import Path
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage


current_dir = Path(__file__).parent  
sys.path.append(str(current_dir))

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
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
    total_duration = timestamps[-1] - timestamps[0]
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

def create_detailed_head_analysis(file_path, save_plot=True, save_csv=False, show_plot=False, global_origin=None):
    positions, rotations, forward_vectors, timestamps = load_complete_head_data(file_path)
    if len(positions) == 0:
        print(f"No valid data found in {file_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {Path(file_path).stem}")
    print(f"{'='*60}")
    
    #Normalizare a pozițiilor
    if global_origin is not None:
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
    
    for combo_name, combo_data in feature_combinations.items():
        cluster_labels = perform_agglomerative_clustering(combo_data, scenario_name, combo_name)
        if cluster_labels is not None:
            np.savez(
                f'clustering_results_{scenario_name}_{combo_name.replace("+", "_")}.npz',
                labels=cluster_labels,
                features=combo_data
            )
            if 'Position' in combo_name:
                plot_clusters(positions, cluster_labels, scenario_name, combo_name)

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
        arrow = Arrow3D([arrow_start[0], arrow_end[0]],
                        [arrow_start[1], arrow_end[1]],
                        [arrow_start[2], arrow_end[2]],
                        mutation_scale=22, lw=2.5, arrowstyle="-|>", color="crimson", alpha=0.7)
        ax.add_artist(arrow)
        arrow_count += 1

    #Marcaje pentru unghiuri (triunghiurile galbene)
    turn_indices = []
    for i in range(2, len(positions)):
        v1 = positions[i-1] - positions[i-2]
        v2 = positions[i] - positions[i-1]
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            if np.degrees(angle) > 30:
                turn_indices.append(i-1)
    
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

def plot_all_scenarios_comparison(all_data, global_origin=None):
    if not all_data:
        print("No data available for plotting.")
        return

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.get_cmap("tab10", len(all_data))

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
                color=colors(idx))

        #start
        ax.scatter(x[0], y[0], z[0], marker='o', color=colors(idx), s=100, edgecolors='black')

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
    plt.show()

def main():
    json_files = list(Path('vr_recordings').glob('*.json'))
    if not json_files:
        print("No JSON files found in vr_recordings directory")
        return

    json_files.sort()
    print(f"Found {len(json_files)} scenario files")

    global_origin = None
    all_data = {}

    for file_path in json_files:
        scenario_name = file_path.stem
        print(f"\nProcessing scenario: {scenario_name}")
        try:
            
            positions, rotations, forward_vectors, timestamps = load_complete_head_data(str(file_path))

            if len(positions) == 0:
                print(f"No valid data for scenario {scenario_name}")
                continue

            
            if global_origin is None:
                global_origin = positions[0]

           
            positions = positions - global_origin

           
            N = 10
            positions = positions[::N]
            rotations = rotations[::N]
            forward_vectors = forward_vectors[::N]
            timestamps = timestamps[::N]

          
            all_data[scenario_name] = np.hstack((
                positions,
                rotations,
                forward_vectors,
                timestamps.reshape(-1, 1)
            ))

          
            create_detailed_head_analysis(file_path, save_plot=True, global_origin=global_origin, show_plot=False)


        except Exception as e:
            print(f"Error processing {scenario_name}: {e}")

    #Cluster global pt toate scenariile
    if all_data:
        combined_data = np.vstack(list(all_data.values()))

        scaler = StandardScaler()
        scaled_combined = scaler.fit_transform(combined_data)
        combined_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
        combined_labels = combined_clustering.fit_predict(scaled_combined)

        np.savez('global_clustering_results.npz',
                 labels=combined_labels,
                 features=combined_data,
                 scenario_names=np.array(list(all_data.keys())))

        try:
            from visual_scenario_analysis import (
                plot_global_scenario_comparison,
                plot_scenario_similarity_heatmap,
                cluster_scenarios_by_mean_vectors
            )
            plot_global_scenario_comparison(all_data)
            plot_scenario_similarity_heatmap(all_data)
            cluster_scenarios_by_mean_vectors(all_data)
        except ImportError:
            print("visual_scenario_analysis module not found.")
            return

        #Comparare 3D pentru toate scenariile
        plot_all_scenarios_comparison(all_data, global_origin=global_origin)

if __name__ == "__main__":
    main()
