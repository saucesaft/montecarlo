from map import PIXELS_PER_METER, MAP_SIZE

def initialize_particles(m, num_particles, xp):
    free_y, free_x = xp.where(m == 0)

    random_indices = xp.random.choice(len(free_x), size=num_particles, replace=True)

    x_coords = free_x[random_indices] + xp.random.uniform(0, 1, size=num_particles)
    y_coords = free_y[random_indices] + xp.random.uniform(0, 1, size=num_particles)

    thetas = xp.random.uniform(0, 2 * xp.pi, size=num_particles)

    return xp.column_stack((x_coords, y_coords, thetas))

def score_particles(particles, lidar, map_array, xp):
    L_x = lidar[0] * PIXELS_PER_METER   # (N_lidar,)
    L_y = lidar[1] * PIXELS_PER_METER

    p_x     = particles[:, 0]           # (N_particles,)
    p_y     = particles[:, 1]
    p_theta = particles[:, 2]

    cos_t = xp.cos(p_theta)[:, None]    # (N_particles, 1)
    sin_t = xp.sin(p_theta)[:, None]

    # rotate each lidar point by each particle's angle, then translate — (N_particles, N_lidar)
    g_x = L_x[None, :] * cos_t - L_y[None, :] * sin_t + p_x[:, None]
    g_y = L_x[None, :] * sin_t + L_y[None, :] * cos_t + p_y[:, None]

    pixel_x = xp.round(g_x).astype(int)
    pixel_y = xp.round(g_y).astype(int)

    valid = (pixel_x >= 0) & (pixel_x < MAP_SIZE) & (pixel_y >= 0) & (pixel_y < MAP_SIZE)

    px_clamped = xp.clip(pixel_x, 0, MAP_SIZE - 1)
    py_clamped = xp.clip(pixel_y, 0, MAP_SIZE - 1)

    hits = map_array[py_clamped, px_clamped] == 1

    return xp.sum(hits & valid, axis=1)

def filter_particles(particles, scores, keep_fraction, xp):
    num_to_keep = int( len(particles) * keep_fraction )

    sorted_indices = xp.argsort(scores)[::-1]

    best_indices = sorted_indices[:num_to_keep]

    best_particles = particles[best_indices]

    # best_scores = scores[best_indices]

    return best_particles
