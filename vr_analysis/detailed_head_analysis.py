import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.cm as cm

#sageti pentru a arata directia in care se uita subiectul
#se creeaza o sageata 3D cu coordonatele de start si end
#se converteste in 2D
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
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
    
    total_duration = timestamps[-1] - timestamps[0]
    if total_duration <= 20:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    #ignorare primele si ultimele 10 secunde
    start_time = timestamps[0] + 10.0
    end_time = timestamps[-1] - 10.0
    valid_indices = (timestamps >= start_time) & (timestamps <= end_time)
    
    return positions[valid_indices], rotations[valid_indices], forward_vectors[valid_indices], timestamps[valid_indices]


def create_detailed_head_analysis(file_path, save_plot=True):

    positions, rotations, forward_vectors, timestamps = load_complete_head_data(file_path)

    if len(positions) == 0:
        print(f"No valid data found in {file_path}")
        return
    
    #setare punct de start setata la acelasi punct pentru toate scenariile
    positions = positions - positions[0] + [2.0, 1.6, 2.0] 

    
    N = 10  #vizualizare la fiecare N puncte (claritate)
    positions = positions[::N]
    forward_vectors = forward_vectors[::N]
    timestamps = timestamps[::N]

    #configurare grafic dimensiune si model 3d
    fig = plt.figure(figsize=(22, 8))
    ax = fig.add_subplot(111, projection='3d')

    #inlocuire Y cu Z
    x = positions[:, 0]
    y = positions[:, 2]  
    z = positions[:, 1]  

    # Plot the 3D route as a bright blue line
    ax.plot(x, y, z,
            color='deepskyblue', linewidth=2.5, alpha=0.95, linestyle='-', label='Head Route')

    #punct de start
    ax.scatter(x[0], y[0], z[0],
               c='lime', s=220, label='Start', marker='o', edgecolors='black', linewidth=3, zorder=10)
    
    #punct de final
    ax.scatter(x[-1], y[-1], z[-1],
               c='red', s=220, label='End', marker='*', edgecolors='black', linewidth=3, zorder=10)

    #adaugare sageti pentru directia de privire
    step = max(1, len(positions) // 25)
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

    #marcarea schimbarilor de directie
    turn_indices = []
    for i in range(2, len(positions)):
        v1 = positions[i-1] - positions[i-2]
        v2 = positions[i] - positions[i-1]
        #calcularea unghiului dintre vectori
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            if np.degrees(angle) > 30:
                turn_indices.append(i-1)
    #personalizare marcaje pentru schimbari de directie
    if turn_indices:
        ax.scatter(x[turn_indices], y[turn_indices], z[turn_indices],
                   c='orange', s=80, marker='^', label='Turns', zorder=9)

    #configurare axele si titlu
    scenario_name = Path(file_path).stem
    ax.set_xlabel('X Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Z Position', fontsize=14, fontweight='bold')  
    ax.set_zlabel('Y Position', fontsize=14, fontweight='bold')
    ax.set_title(f'3D Head Movement Route - {scenario_name}', fontsize=18, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)

    ax.view_init(elev=30, azim=120)

    plt.tight_layout()

    if save_plot:
        output_file = f"detailed_analysis_{scenario_name}_3d.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D route saved as {output_file}")

    plt.show()

def main():

    #citire fisiere json din directorul vr_recordings
    json_files = list(Path('vr_recordings').glob('*.json'))
    
    if not json_files:
        print("No JSON files found in vr_recordings directory")
        return
    
    #sortare fisiere dupa nume
    json_files.sort()
    print(f"Found {len(json_files)} scenario files")
    
    #procesare fiecare fisier json
    for file_path in json_files:
        scenario_name = file_path.stem
        print(f"\nProcessing scenario: {scenario_name}")
        
        try:
            create_detailed_head_analysis(str(file_path))
        except Exception as e:
            print(f"Error processing {scenario_name}: {e}")

if __name__ == "__main__":
    main() 