import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

import map
import particles as pf

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

MAP = xp.asarray(map._data)
MAP_SIZE = map.MAP_SIZE
PIXELS_PER_METER = map.PIXELS_PER_METER

NUM_PARTICLES = 1000

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
    
    mask_lower = scan_data >= map.LOWER_BOUND
    mask_upper = scan_data <= map.UPPER_BOUND

    filtered_distance = scan_data[mask_lower & mask_upper]

    if len(filtered_distance) == 0:
        return None

    degrees = xp.array(range(len(scan_data)))
    degrees = xp.radians(degrees)

    degrees_filtered = degrees[mask_lower & mask_upper]

    x = filtered_distance * xp.cos(degrees_filtered) * -1
    y = filtered_distance * xp.sin(degrees_filtered) * -1

    grid_center = MAP_SIZE // 2
    grid_x = xp.round(x * PIXELS_PER_METER).astype(int) + grid_center
    grid_y = xp.round(y * PIXELS_PER_METER).astype(int) + grid_center

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

def delta_movement(x_vel, theta_vel, dt):
    delta_x     = x_vel * dt 
    delta_theta = theta_vel * dt

    return np.array([delta_x, delta_theta])

if __name__ == "__main__":
    cameras_to_use = StretchCameras.none()

    sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)

    sim.start(headless=False)

    particles = pf.initialize_particles(MAP, NUM_PARTICLES, xp)

    try:
        sim.set_base_velocity(v_linear=5.0, omega=30)

        target = 1.1
        
        dt = time.perf_counter()
        prev_time = 0

        while sim.is_running():
            status = sim.pull_status()
            sensor_data = sim.pull_sensor_data()

            try:
                # calculate dt
                current_time = time.perf_counter()
                dt = current_time - prev_time
                prev_time = current_time

                # simulate 2D lidar
                data = laser_scan( scan_data=sensor_data.get_data(StretchSensors.base_lidar), particles=particles )
                
                # plot lidar with matplotlib
                plot_map_particles( particles )
                
                # TODO visualize LiDAR at (0,0)
                # TODO visualize LiDAR at most probable particle
                # TODO visualize dt

                if data is None:
                    continue

                ## compute scores for each particles using sums with the map ##
                scores = pf.score_particles(particles, data, MAP, xp)
                
                ## keep 20% percent of top best particles that match ##
                bp = pf.filter_particles(particles, scores, 0.20, xp)

                # base x_vel and theta_vel is calculated from the mujoco kinematics
                dv = delta_movement(status.base.x_vel, status.base.theta_vel, dt)

                # convert to pixel's ratio (but not theta)
                dvp_x = dv[0] * map.PIXELS_PER_METER 
                dvp_theta = dv[1]

                bp[:, 0] += dvp_x * xp.cos( bp[:, 2] )
                bp[:, 1] += dvp_x * xp.sin( bp[:, 2] )
                bp[:, 2] += dvp_theta

                ## resmaple particles from past ones ##
                indices = xp.random.choice(len(bp), size=NUM_PARTICLES, replace=True)

                particles = bp[indices]

                ## add jitter ##
                # if we don't add this, resampling leads to duplicate high-score particles which collapse to a single dot
                particles[:, 0] += xp.random.normal(0, 1.5, size=NUM_PARTICLES)   # pixels
                particles[:, 1] += xp.random.normal(0, 1.5, size=NUM_PARTICLES)
                particles[:, 2] += xp.random.normal(0, 0.05, size=NUM_PARTICLES)  # radians

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
