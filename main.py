import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

import map

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="Use GPU accelerated version")
args = parser.parse_args()

if args.gpu:
    import cupy as xp
else:
    import numpy as xp

try:
    matplotlib.use("TkAgg")
    plt.ion()
except: ...

MAP = map.load( "map.png" )
MAP_SIZE = MAP.shape[0]

NUM_PARTICLES = 1000
PARTICLES = xp.zeros((NUM_PARTICLES, 3))

### particle filter
def initialize_particles(m, num_particles):
    free_y, free_x = xp.where(m == 0)
    
    random_indices = xp.random.choice(len(free_x), size=num_particles, replace=True)
    
    x_coords = free_x[random_indices] + xp.random.uniform(0, 1, size=num_particles)
    y_coords = free_y[random_indices] + xp.random.uniform(0, 1, size=num_particles)
    
    thetas = xp.random.uniform(0, 2 * xp.pi, size=num_particles)
    
    particles = xp.column_stack((x_coords, y_coords, thetas))
    
    return particles

### kinematics ###
def forward_kinematics(w_l, w_r, r, L):
    mat = xp.array(
                    [ [  r/2, r/2 ],
                      [ -r/L, r/L ] ]
                  )

    wheel_speeds = xp.array( [ w_l, w_r ] )

    return mat @ wheel_speeds

def inverse_kinematics(v, omega, r, L):

    mat = xp.array(
                    [ [ 1/r, -L/(2*r)],
                      [ 1/r,  L/(2*r)] ]
                  )

    body_speed = xp.array( [v, omega] )

    return mat @ body_speed


def laser_scan(scan_data: np.ndarray, particles: xp.ndarray):
    # convert simulator numpy data to current backend (xp)
    scan_data = xp.asarray(scan_data)
    
    lower_bound = 0.2
    upper_bound = 5

    mask_lower = scan_data >= lower_bound
    mask_upper = scan_data <= upper_bound

    filtered_distance = scan_data[mask_lower & mask_upper]

    if len(filtered_distance) == 0:
        return None

    degrees = xp.array(range(len(scan_data)))
    degrees = xp.radians(degrees)

    degrees_filtered = degrees[mask_lower & mask_upper]

    x = filtered_distance * xp.cos(degrees_filtered) * -1
    y = filtered_distance * xp.sin(degrees_filtered) * -1

    # determinar la resolucion
    # - el diametro son 10 metros (upper bound)
    pixels_per_meter = (MAP_SIZE / (upper_bound * 2)) 

    grid_center = MAP_SIZE // 2
    grid_x = xp.round(x * pixels_per_meter).astype(int) + grid_center
    grid_y = xp.round(y * pixels_per_meter).astype(int) + grid_center

    valid_mask = (grid_x >= 0) & (grid_x < MAP_SIZE) & (grid_y >= 0) & (grid_y < MAP_SIZE)
    valid_x = grid_x[valid_mask]
    valid_y = grid_y[valid_mask]

    return xp.array( [x, y] )

def plot_map_particles(particles: xp.ndarray):
    # move data to CPU for matplotlib
    plot_map = MAP.get() if args.gpu else MAP
    plot_particles = particles.get() if args.gpu else particles

    plt.imshow(plot_map, cmap='binary', origin='lower')

    p_x = plot_particles[:, 0]
    p_y = plot_particles[:, 1]

    plt.scatter(p_x, p_y, color='red', s=2, alpha=0.3)

    grid_center = MAP_SIZE // 2

    plt.axhline(grid_center, color='r', linestyle='--', alpha=0.3)
    plt.axvline(grid_center, color='r', linestyle='--', alpha=0.3)
    plt.title("grid map")

    plt.pause(0.001)
    plt.cla()

if __name__ == "__main__":
    cameras_to_use = StretchCameras.none()

    sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)

    sim.start(headless=False)

    particles = initialize_particles( MAP, NUM_PARTICLES )

    try:
        sim.set_base_velocity(v_linear=5.0, omega=30)

        target = 1.1  # m
        while sim.is_running():
            status = sim.pull_status()
            sensor_data = sim.pull_sensor_data()

            try:
                data = laser_scan( scan_data=sensor_data.get_data(StretchSensors.base_lidar), particles=particles )
                
                plot_map_particles( particles )

            except Exception as e:
                print( e ) 

            current_position = status.base.x

            if target > 0 and current_position > target:
                target *= -1
                sim.set_base_velocity(v_linear=-5.0, omega=-30)
            elif target < 0 and current_position < target:
                target *= -1
                sim.set_base_velocity(v_linear=5.0, omega=30)

    except KeyboardInterrupt:
        sim.stop()
