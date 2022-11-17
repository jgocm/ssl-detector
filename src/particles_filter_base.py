import numpy as np
import math
from jetson_vision import JetsonVision
from particles_resampler import Resampler
from particle_vision import ParticleVision

class Particle:
    '''
    Particle pose has 3 degrees of freedom:
        x: particle position towards the global X axis
        y: particle position towards the global Y axis
        theta: particle orientation towards the field axis     

    State: 
        (x, y, theta)
    
    Constraints:
        is_out_of_field: returns if the particle is out-of-field boundaries
    '''
    def __init__(
                self,
                initial_state = [0,0,0],
                weight = 1
                ):
        self.state = initial_state
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = ((self.state[2] + 180) % 360) - 180
        self.weight = weight
        self.vision = ParticleVision()

    def from_weighted_sample(self, sample):
        self = Particle(initial_state=sample[1], weight=sample[0])

    def as_weighted_sample(self):
        return [self.weight,[self.x, self.y, self.theta]]

    def is_out_of_field(self, field):
        '''
        Check if particle is out of field boundaries
        
        param: current field configurations
        return: True if particle is out of field boundaries
        '''
        if self.x > field.x_max:
            return True
        elif self.y > field.y_max:
            return True
        elif self.x < field.x_min:
            return True
        elif self.y < field.y_min:
            return True
        else:
            return False

    def move(self, movement):
        self.state = [sum(x) for x in zip(self.state, movement)]
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = ((self.state[2] + 180) % 360) - 180

class ParticleFilter:
    def __init__(
                self,
                number_of_particles,
                field,
                process_noise,
                measurement_noise
                ):

        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))
        
        # Initialize filter settings
        self.n_particles = number_of_particles
        self.particles = []

        # Field related settings
        self.state_dimension = len(Particle().state)
        self.field = field

        # Set noise
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Resampling
        self.resampler = Resampler()

    def initialize_particles_from_seed_position(self, seed_x, seed_y, max_distance):
        """
        Initialize the particles uniformly around a seed position (x, y, orientation). 
        """
        particles = []
        weight = 1.0/self.n_particles
        for i in range(self.n_particles):
            radius = np.random.uniform(0, max_distance)
            direction = np.random.uniform(0, 360)
            orientation = np.random.uniform(0, 360)
            x = seed_x + radius*math.cos(direction)
            y = seed_y + radius*math.sin(direction)
            particle = Particle(initial_state=[x, y, orientation], weight=weight)
            particles.append(particle)
        
        self.particles = particles
        
    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, orientation). 
        No arguments are required and function always succeeds hence no return value.
        """

        # Initialize particles with uniform weight distribution
        particles = []
        weight = 1.0 / self.n_particles
        for i in range(self.n_particles):
            particle = Particle(
                initial_state=[
                    np.random.uniform(self.field.x_min, self.field.x_max),
                    np.random.uniform(self.field.y_min, self.field.y_max),
                    np.random.uniform(-180, 180)],
                weight=weight)

            particles.append(particle)
        
        self.particles = particles

    def initialize_particles_gaussian(self, mean_vector, standard_deviation_vector):
        """
        Initialize particle filter using a Gaussian distribution with dimension three: x, y, orientation. 
        Only standard deviations can be provided hence the covariances are all assumed zero.

        :param mean_vector: Mean of the Gaussian distribution used for initializing the particle states
        :param standard_deviation_vector: Standard deviations (one for each dimension)
        :return: Boolean indicating success
        """

        # Check input dimensions
        if len(mean_vector) != self.state_dimension or len(standard_deviation_vector) != self.state_dimension:
            print("Means and state deviation vectors have incorrect length in initialize_particles_gaussian()")
            return False

        # Initialize particles with uniform weight distribution
        self.particles = []
        weight = 1.0 / self.n_particles
        for i in range(self.n_particles):
            initial_state = np.random.normal(mean_vector, standard_deviation_vector, self.state_dimension).tolist()
            particle = Particle(initial_state=initial_state, weight=weight)
            while particle.is_out_of_field(self.field):
                # Get state sample
                initial_state = np.random.normal(mean_vector, standard_deviation_vector, self.state_dimension).tolist()
                particle = Particle(initial_state=initial_state, weight=weight)

            # Add particle i
            self.particles.append(particle)

    def particles_as_weigthed_samples(self):
        samples = []
        for particle in self.particles:
            samples.append(particle.as_weighted_sample())
        return samples

    def get_average_state(self):
        """
        Compute average state according to all weighted particles

        :return: Average x-position, y-position and orientation
        """

        # Compute sum of all weights
        sum_weights = 0.0
        for particle in self.particles:
            sum_weights += particle.weight

        # Compute weighted average
        avg_x = 0.0
        avg_y = 0.0
        avg_theta = 0.0
        for particle in self.particles:
            avg_x += particle.x / sum_weights * particle.weight
            avg_y += particle.y / sum_weights * particle.weight
            avg_theta += particle.theta / sum_weights * particle.weight

        return [avg_x, avg_y, avg_theta]

    def get_max_weight(self):
        """
        Find maximum weight in particle filter.

        :return: Maximum particle weight
        """
        return max([particle.as_weigthed_sample()[0] for particle in self.particles])

    def normalize_weights(self, weights):
        """
        Normalize all particle weights.
        """
        # Compute sum weighted samples
        sum_weights = sum(weights)     

        # Check if weights are non-zero
        if sum_weights < 1e-15:
            print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

            # Set uniform weights
            return [(1.0 / len(weights)) for i in weights]

        # Return normalized weights
        return [weight / sum_weights for weight in weights]

    def propagate_particles(self, movement):
        """
        Propagate particles from odometry movement measurements. 
        Return the propagated particle.

        :param movement: [forward motion, side motion and rotation] in meters and radians
        """
        # TODO: Add noise
        
        # Move particles
        for particle in self.particles:
            particle.move(movement)

            if particle.is_out_of_field(self.field):
                # print("Particle Out of Field Boundaries")
                particle.weight = 0        

    def compute_observation(self, particle):
        boundary_points = particle.vision.detect_boundary_points(
                                    particle.x, 
                                    particle.y, 
                                    particle.theta, 
                                    self.field)
        print(f'weight: {particle.weight}, position:{particle.state}, detection:{boundary_points}')
        return boundary_points

    def compute_likelihood(self, measurements, particle):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample observations.

        :param measurements: Current measurements
        :param observations: Detected wall relative positions from the sample vision
        :return Likelihood
        """
        # Check if particle is out of field boundaries
        if particle.is_out_of_field(self.field):
            return 0
        
        else:
            # Initialize measurement likelihood
            likelihood_sample = 1.0
            
            # Compute particle observations
            observations = self.compute_observation(particle)
            # Compute difference between real measurements and sample observations
            differences = np.array(measurements) - observations
            # Loop over all observations for current particle
            for diff in differences:
                # Map difference true and expected angle measurement to probability
                p_z_given_x = \
                    np.exp(-(diff[0]) * (diff[0]) /
                        (2 * self.measurement_noise[0] * self.measurement_noise[0]))

                p_z_given_y = \
                    np.exp(-(diff[1]) * (diff[1]) /
                        (2 * self.measurement_noise[1] * self.measurement_noise[1]))

                # Incorporate likelihoods current landmark
                likelihood_sample *= p_z_given_x * p_z_given_y
                if likelihood_sample<1e-15:
                    return 0

            # Return importance weight based on all landmarks
            return likelihood_sample

    def needs_resampling(self):
        '''
        TODO: implement method for checking if resampling is needed
        '''
        for particle in self.particles:
            if particle.weight>0.9:
                return True
        
        else: return False

    def update(self, movement, measurements):
        """
        Process a measurement given the measured robot displacement and resample if needed.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """
        weights = []
        for particle in self.particles:
            # Compute current particle's weight based on likelihood
            weight = particle.weight * self.compute_likelihood(measurements, particle)
            # Store weight for normalization
            weights.append(weight)

        # Update to normalized weights
        weights = self.normalize_weights(weights)
        for i in range(self.n_particles):
            self.particles[i].weight = weights[i]
        
        # Resample if needed
        if self.needs_resampling():
            mean_state = self.get_average_state()
            self.initialize_particles_gaussian(mean_vector=mean_state, standard_deviation_vector=[0.1, 0.1, 30])
            # samples = self.resampler.resample(
            #                 self.particles_as_weigthed_samples(), 
            #                 self.n_particles, 
            #                 self.resampling_algorithm)
            # for i in range(self.n_particles):
            #     self.particles[i].from_weighted_sample(samples[i])

        # Propagate the particles state according to the current movement
        self.propagate_particles(movement) 


if __name__ == "__main__":
    from entities import Field
    from jetson_vision import JetsonVision
    import time
    import os
    import cv2
    from glob import glob

    cwd = os.getcwd()

    # SET EMBEDDED VISION
    vision = JetsonVision(vertical_lines_offset=320, 
                        debug=True)

    # SET FIELD DIMENSIONS
    field = Field()
    field.redefineFieldLimits(x_max=4, y_max=3, x_min=-0.5, y_min=-3)

    # INITIALIZE PARTICLES FILTER
    robot_tracker = ParticleFilter(
            number_of_particles=5,
            field=field,
            process_noise = [1, 1, 1],
            measurement_noise = [1, 1])


    # FILE COUNTERS
    # first squared path starts moving on frame 137
    frame_nr = 134
    quadrado_nr = 1

    # INITIAL POSITION
    initial_position_dir = glob(cwd + f"/data/quadrado{quadrado_nr}/1_*.txt")
    initial_position = np.loadtxt(initial_position_dir[0])
    seed_x, seed_y, seed_radius = initial_position[0], initial_position[1], 0.5
    robot_tracker.initialize_particles_from_seed_position(seed_x, seed_y, seed_radius)

    while frame_nr<500:
        # LOAD FRAME
        WINDOW_NAME = "PARTICLES FILTER DEVELOPMENT"
        frame_dir = cwd + f"/data/quadrado{quadrado_nr}/{frame_nr}_*.jpg"
        file = glob(frame_dir)
        img = cv2.imread(file[-1])
        # height, width = img.shape[0], img.shape[1]

        # LOAD ODOMETRY DATA
        if frame_nr>134:
            last_position = current_position
        else:
            last_position = initial_position
        odometry_dir = cwd + f"/data/quadrado{quadrado_nr}/{frame_nr}_*.txt"
        file = glob(odometry_dir)
        current_position = np.loadtxt(file[-1])
        movement = current_position - last_position
        
        # MAKE VISION OBSERVATION
        _, _, _, _, particle_filter_observations = vision.process(img, timestamp=time.time())
        boundary_ground_points, line_ground_points = particle_filter_observations

        # SHOW ON SCREEN
        cv2.imshow(WINDOW_NAME, img)

        # UPDATE PARTICLES FILTER
        print(f'odometry: {current_position}, observation: {boundary_ground_points}')
        if len(boundary_ground_points)>0:
            robot_tracker.update(movement, boundary_ground_points)

        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            break
        else:
            frame_nr=frame_nr+1



