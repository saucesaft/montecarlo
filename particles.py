from map import PIXELS_PER_METER, MAP_SIZE

def initialize_particles(m, num_particles, xp):
    free_y, free_x = xp.where(m == 0)

    random_indices = xp.random.choice(len(free_x), size=num_particles, replace=True)

    x_coords = free_x[random_indices] + xp.random.uniform(0, 1, size=num_particles)
    y_coords = free_y[random_indices] + xp.random.uniform(0, 1, size=num_particles)

    thetas = xp.random.uniform(0, 2 * xp.pi, size=num_particles)

    return xp.column_stack((x_coords, y_coords, thetas))

# TODO vectorize this function
def score_particles(particles, lidar, map_array, xp):

    local_lidar_x = lidar[0]
    local_lidar_y = lidar[1]

    scores = xp.zeros( len(particles) ) # initialize to zeros our scores for each particle

    # convert lidar distances to pixels
    L_x = local_lidar_x * PIXELS_PER_METER 
    L_y = local_lidar_y * PIXELS_PER_METER

    # we basically want to iterate through all the particles
    # the score depends on how much does our current reading matches the map
    # to do this, we go and test on all particles,
    # the more the points match, the higher score we have
    
    for i in range( len(particles) ):
        p_x = particles[i, 0]
        p_y = particles[i, 1]
        p_theta = particles[i, 2]

        score = 0

        for j in range( len(L_x) ):

            # the actual lidar point
            l_x = L_x[j]
            l_y = L_y[j]

            # rotate the local point by the particle's angle
            # then translate (add) the particle's global position
            g_x = (l_x * xp.cos(p_theta)) - (l_y * xp.sin(p_theta)) + p_x
            g_y = (l_x * xp.sin(p_theta)) + (l_y * xp.cos(p_theta)) + p_y

            # use int to get exact matrix indices
            pixel_x = int(xp.round(g_x))
            pixel_y = int(xp.round(g_y))

            # are we inside the map dimensions
            if 0 <= pixel_x < MAP_SIZE and 0 <= pixel_y < MAP_SIZE:
                if map_array[pixel_y, pixel_x] == 1:
                    score += 1

        scores[i] = score

    return scores
