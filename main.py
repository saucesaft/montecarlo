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

NUM_PARTICLES = 1024

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


def laser_scan(scan_data: xp.ndarray, particles: xp.ndarray):
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
    
    x = filtered_distance * xp.cos(degrees_filtered)
    y = filtered_distance * xp.sin(degrees_filtered + xp.pi)

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

    plt.imshow(plot_map, cmap='binary', origin='upper')

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

    return xp.array([delta_x, delta_theta])

if __name__ == "__main__":
    cameras_to_use = StretchCameras.none()

    sim = StretchMujocoSimulator(cameras_to_use=cameras_to_use)

    sim.start(headless=False)

    particles = pf.initialize_particles(MAP, NUM_PARTICLES, xp)

    try:
        # each waypoint: (x_target, v_linear, omega) — varied curvatures break map symmetry
        waypoints = [
            ( 1.2,  5.0,  15),
            (-0.4,  5.0,  45),
            ( 1.0,  5.0,  55),
            (-1.1,  5.0, -20),
            ( 0.3,  5.0,  40),
            (-0.9, -5.0, -60),
        ]
        wp_idx = 0
        sim.set_base_velocity(v_linear=waypoints[0][1], omega=waypoints[0][2])

        LOOP_HZ = 20
        LOOP_DT = 1.0 / LOOP_HZ

        while sim.is_running():
            loop_start = time.perf_counter()

            status = sim.pull_status()
            sensor_data = sim.pull_sensor_data()

            x_target, v, omega = waypoints[wp_idx]
            if (v > 0 and status.base.x >= x_target) or (v < 0 and status.base.x <= x_target):
                wp_idx = (wp_idx + 1) % len(waypoints)
                _, v, omega = waypoints[wp_idx]
                sim.set_base_velocity(v_linear=v, omega=omega)

            try:
                dt = LOOP_DT

                # simulate 2D lidar
                data = laser_scan( scan_data=sensor_data.get_data(StretchSensors.base_lidar), particles=particles )

                # plot lidar with matplotlib
                plot_map_particles( particles )

                # TODO visualize LiDAR at (0,0)
                # TODO visualize LiDAR at most probable particle
                # TODO visualize dt

                if data is None:
                    continue

                # predict: move particles forward before scoring
                dv = delta_movement(status.base.x_vel, status.base.theta_vel, dt)
                dvp_x = dv[0] * map.PIXELS_PER_METER

                particles[:, 0] += dvp_x * xp.cos( particles[:, 2] )
                particles[:, 1] += dvp_x * xp.sin( particles[:, 2] )
                particles[:, 2] += dv[1]

                ## update: score then resample ##
                scores = pf.score_particles(particles, data, MAP, xp)
                particles = pf.resample_particles(particles, scores, NUM_PARTICLES, xp)

                ## add jitter ##
                # if we don't add this, resampling leads to duplicate high-score particles which collapse to a single dot
                particles[:, 0] += xp.random.normal(0, 1.5, size=NUM_PARTICLES)   # pixels
                particles[:, 1] += xp.random.normal(0, 1.5, size=NUM_PARTICLES)
                particles[:, 2] += xp.random.normal(0, 0.05, size=NUM_PARTICLES)  # radians

            except Exception as e:
                print( e )

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, LOOP_DT - elapsed))

    except KeyboardInterrupt:
        sim.stop()
