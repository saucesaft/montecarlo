import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

try:
    # Some machines seem to need this for matplotlib to work.
    matplotlib.use("TkAgg")
except: ...

MAP_SIZE = 64

MAP = np.zeros(( MAP_SIZE, MAP_SIZE ))

NUM_PARTICLES = 1000
PARTICLES = np.zeros((NUM_PARTICLES, 3))

def initialize_particles(m, num_particles):
    free_y, free_x = np.where(m== 0)
    
    random_indices = np.random.choice(len(free_x), size=num_particles, replace=True)
    
    x_coords = free_x[random_indices] + np.random.uniform(0, 1, size=num_particles)
    y_coords = free_y[random_indices] + np.random.uniform(0, 1, size=num_particles)
    
    thetas = np.random.uniform(0, 2 * np.pi, size=num_particles)
    
    particles = np.column_stack((x_coords, y_coords, thetas))
    
    return particles

### kinematics ###
def forward_kinematics(w_l, w_r, r, L):
    mat = np.array(
                    [ [  r/2, r/2 ],
                      [ -r/L, r/L ] ]
                  )

    wheel_speeds = np.array( [ w_l, w_r ] )

    return mat @ wheel_speeds

def inverse_kinematics(v, omega, r, L):

    mat = np.array(
                    [ [ 1/r, -L/(2*r)],
                      [ 1/r,  L/(2*r)] ]
                  )

    body_speed = np.array( [v, omega] )

    return mat @ body_speed


def show_laser_scan(scan_data: np.ndarray, particles: np.ndarray):

    lower_bound = 0.2
    upper_bound = 5

    mask_lower = scan_data >= lower_bound
    mask_upper = scan_data <= upper_bound

    filtered_distance = scan_data[mask_lower & mask_upper]
    # filtered_distance = scan_data

    if len(filtered_distance) == 0:
        return time.sleep(1 / 15)

    degrees = np.array(range(len(scan_data)))

    degrees = np.radians(degrees)

    # print( degrees )
    
    degrees_filtered = degrees[mask_lower & mask_upper]

    x = filtered_distance * np.cos(degrees_filtered) * -1
    y = filtered_distance * np.sin(degrees_filtered) * -1

    # determinar la resolucion
    # - el diametro son 10 metros (upper bound)
    pixels_per_meter = (MAP_SIZE / (upper_bound * 2)) * 3

    grid_center = MAP_SIZE // 2
    grid_x = np.round(x * pixels_per_meter).astype(int) + grid_center
    grid_y = np.round(y * pixels_per_meter).astype(int) + grid_center

    valid_mask = (grid_x >= 0) & (grid_x < MAP_SIZE) & (grid_y >= 0) & (grid_y < MAP_SIZE)
    valid_x = grid_x[valid_mask]
    valid_y = grid_y[valid_mask]

    MAP = np.zeros((MAP_SIZE, MAP_SIZE))
    MAP[valid_y, valid_x] = 1

    plt.imshow(MAP, cmap='binary', origin='lower')
    
    p_x = particles[:, 0]
    p_y = particles[:, 1]
    plt.scatter(p_x, p_y, color='red', s=2, alpha=0.3)

    plt.axhline(grid_center, color='r', linestyle='--', alpha=0.3)
    plt.axvline(grid_center, color='r', linestyle='--', alpha=0.3)
    plt.title("Occupancy Grid Map")

    degrees_filtered = np.rad2deg(degrees)

    # scan_data = np.clip(scan_data, a_min=0.2, a_max=5)

    # plt.bar( degrees, scan_data )

    # front_idx = (degrees_filtered >= 150) & (degrees_filtered <= 210) # ~180
    # back_idx = (degrees_filtered >= 330) | (degrees_filtered <= 30) # ~0
    # right_idx= (degrees_filtered >= 60) & (degrees_filtered <= 120) # ~90
    # left_idx = (degrees_filtered >= 240) & (degrees_filtered <= 300) # ~270
    #
    # plt.scatter(x, y, color="r", s=5)
    # plt.scatter(x[front_idx], y[front_idx], color="g", s=5)
    # plt.scatter(x[back_idx], y[back_idx], color="b", s=5)
    # plt.scatter(x[left_idx], y[left_idx], color="k", s=5)
    # plt.scatter(x[right_idx], y[right_idx], color="c", s=5)
    # max_x = np.abs(x).max()
    # max_y = np.abs(y).max()
    # plt.xlim([-max_x-1, max_x+1])
    # plt.ylim([-max_y-1, max_y+1])
    # plt.legend(["All", "Front", "Back", "Left", "Right"])

    plt.pause(1 / 15)
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
                show_laser_scan( scan_data=sensor_data.get_data(StretchSensors.base_lidar), particles=particles )
                
                particles = initialize_particles( MAP, NUM_PARTICLES )

            except: ...

            current_position = status.base.x

            if target > 0 and current_position > target:
                target *= -1
                sim.set_base_velocity(v_linear=-5.0, omega=-30)
            elif target < 0 and current_position < target:
                target *= -1
                sim.set_base_velocity(v_linear=5.0, omega=30)

    except KeyboardInterrupt:
        sim.stop()
