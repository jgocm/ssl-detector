import numpy as np
import math
from jetson_vision import JetsonVision
from particles_resampler import Resampler

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

    def from_weighted_sample(self, sample):
        self = Particle(initial_state=sample[1], weight=sample[0])

    def as_weighted_sample(self):
        return [self.weight,[self.x, self.y, self.theta]]

    def is_out_of_field(self, x_max, y_max):
        '''
        Check if particle is out of field boundaries
        
        param: current field configurations
        return: True if particle is out of field boundaries
        '''
        if np.abs(self.x) > x_max:
            return True
        elif np.abs(self.y) > y_max:
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
                measurement_noise,
                vertical_lines_offset,
                resampling_algorithm
                ):

        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))
        
        # Initialize filter settings
        self.n_particles = number_of_particles
        self.particles = []

        # State related settings
        self.state_dimension = len(Particle().state)
        self.x_max = field.length/2 + field.boundary_width
        self.x_min = -self.x_max
        self.y_max = field.width/2 + field.boundary_width
        self.y_min = -self.y_max
        self.field = field

        # Particle sensors
        self.vision = JetsonVision(vertical_lines_offset=vertical_lines_offset)

        # Set noise
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Resampling
        self.resampling_algorithm = resampling_algorithm
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
                    np.random.uniform(self.x_min, self.x_max),
                    np.random.uniform(self.y_min, self.y_max),
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
            while particle.is_out_of_field(x_max=self.x_max, y_max=self.y_max):
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

            if particle.is_out_of_field(x_max=self.x_max, y_max=self.y_max):
                print("Particle Out of Field Boundaries")
                particle.weight = 0

    def compute_observation(self, particle):
        boundary_points = self.vision.detect_boundary_points(
                                    particle.x, 
                                    particle.y, 
                                    particle.theta, 
                                    self.field)
        
        return boundary_points

    def compute_likelihood(self, measurements, particle):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample observations.

        :param measurements: Current measurements
        :param observations: Detected wall relative positions from the sample vision
        :return Likelihood
        """
        # Check if particle is out of field boundaries
        if particle.is_out_of_field(x_max=self.x_max, y_max=self.y_max):
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

        # Propagate the particles state according to the current movements
        self.propagate_particles(movement)

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


if __name__ == "__main__":
    from entities import Ball, Goal, Robot, Field

    # 1. FIX FIELD CONFIGS
    field = Field()