import math
import numpy as np
import pygame

###########
### ANN ###
###########

class ANN:
    def __init__(self, weights1, weights2, input_size=12, hidden_size=4, output_size=2, feedback_delta_t=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feedback_delta_t = feedback_delta_t
        self.previous_output = 0
        self.hidden = np.zeros((self.hidden_size, 1))

        # Initialize weights with random values
        self.weights_ih = weights1
        self.weights_ho = weights2

        # Initialize activation function (ReLU)
        self.relu = lambda x: np.maximum(0, x)
        self.sigmoid = lambda x: 1/(1+ np.exp(-x))


    def feedforward(self, input):
        # Calculate hidden layer output
        self.hidden = self.sigmoid(np.dot(self.weights_ih, np.concatenate((input, self.hidden), axis=None)))

        # Calculate output layer output
        output = np.dot(self.weights_ho, self.hidden)*1

        # Update feedback variables
        self.delta_o = (output - self.previous_output) / self.feedback_delta_t
        self.previous_output = output

        return output
    #
    # def backpropagate(self, input, target):
    #     # Feed input through the network
    #     output = self.feedforward(input)
    #
    #     # Calculate output error and delta
    #     error = target - output
    #     delta = error + self.delta_o
    #
    #     # Calculate hidden layer error and delta
    #     hidden_error = np.dot(self.weights_ho.T, delta)
    #     hidden_delta = hidden_error * (self.relu(np.dot(self.weights_ih, input)) > 0)
    #
    #     # Update weights and biases
    #     self.weights_ho += np.dot(delta, self.relu(np.dot(self.weights_ih, input)).T)
    #     self.weights_ih += np.dot(hidden_delta, input.T)
    #
    #
    # def train(self, inputs, targets, epochs):
    #     for epoch in range(epochs):
    #         for i in range(len(inputs)):
    #             input = inputs[i].reshape(self.input_size, 1)
    #             target = targets[i].reshape(self.output_size, 1)
    #             self.backpropagate(input, target)

    def predict(self, inputs):
        return self.feedforward(inputs)
        outputs = []
        for input in inputs:
            output = self.feedforward(input.reshape(self.input_size, 1))
            outputs.append(output)
        return np.array(outputs)
###############################################
############ JEAN ############################
###############################################

# Define a function to handle collision with walls
def handle_collision(wall_rect):
    # Calculate the difference between the object and wall positions
    WALL_WIDTH = wall_rect.right - wall_rect.left
    WALL_HEIGHT = wall_rect.bottom - wall_rect.top
    top_point = [object_rect.centerx - wall_rect.centerx, object_rect.centery - wall_rect.centery - OBJECT_HEIGHT // 2 ]
    right_point = [object_rect.centerx- wall_rect.centerx + OBJECT_WIDTH // 2, object_rect.centery - wall_rect.centery]
    left_point = [object_rect.centerx- wall_rect.centerx - OBJECT_WIDTH // 2, object_rect.centery - wall_rect.centery]
    bottom_point = [object_rect.centerx- wall_rect.centerx, object_rect.centery - wall_rect.centery + OBJECT_HEIGHT // 2 ]

    if abs(top_point[0]) < WALL_WIDTH // 2 and abs(top_point[1]) < WALL_HEIGHT // 2:
        object_rect.top = wall_rect.bottom
    elif abs(bottom_point[0]) < WALL_WIDTH // 2 and abs(bottom_point[1]) < WALL_HEIGHT // 2:
        object_rect.bottom = wall_rect.top
    elif abs(right_point[0]) < + WALL_WIDTH // 2 and abs(right_point[1]) <  WALL_HEIGHT // 2:
        object_rect.right = wall_rect.left
    elif abs(left_point[0]) < + WALL_WIDTH // 2 and abs(left_point[1]) < + WALL_HEIGHT // 2:
        object_rect.left = wall_rect.right


def handle_outside_collision():
    object_rect.top = max(0, object_rect.top)
    object_rect.bottom = min(SCREEN_HEIGHT, object_rect.bottom)
    object_rect.right = min(SCREEN_WIDTH, object_rect.right)
    object_rect.left = max(0, object_rect.left)



def sensor_calc():
    if object_rect.centerx > wall_rect.left and object_rect.centerx < wall_rect.right:
        if object_rect.top > wall_rect.centery:
            # object is below wall
            RL =  [object_rect.top - wall_rect.bottom, SCREEN_HEIGHT - object_rect.bottom, SCREEN_WIDTH - object_rect.right,
                    object_rect.left]
        else:
            # object is above wall
            RL = [object_rect.top, wall_rect.top - object_rect.bottom, SCREEN_WIDTH - object_rect.right,
                    object_rect.left]
    elif object_rect.centery > wall_rect.top and object_rect.centery < wall_rect.bottom:
        if object_rect.right < wall_rect.centerx:
            #object is to the left of wall
            RL = [object_rect.top, SCREEN_HEIGHT - object_rect.bottom, wall_rect.left - object_rect.right,
                    object_rect.left]
        else:
            #object is to the right of wall
            RL = [object_rect.top, SCREEN_HEIGHT - object_rect.bottom, SCREEN_WIDTH - object_rect.right,
                    object_rect.left - wall_rect.right]
    else:
        RL = [object_rect.top, SCREEN_HEIGHT - object_rect.bottom, SCREEN_WIDTH - object_rect.right,
                    object_rect.left]

    for i in range(4):
        RL[i] = min(200, RL[i])
    return RL
###############################################
############ JAN ##############################
###############################################
def circle_sensor_calc(rotation, wallList):

    #rotation = rotation/180 * math.pi
    sensor_positions = []
    sensor_reading = []
    for i in range(12):
        value = ((2*i/12*math.pi) + rotation)
        sensor_positions.append(value % (2*math.pi))

    #print(sensor_positions)

    scenario = -1
    AllSensorVal = []
    for wall_rect in wallList:
        if object_rect.centerx > wall_rect.left and object_rect.centerx < wall_rect.right:
            if object_rect.top > wall_rect.centery: # object is below wall
                s_range = [.5*math.pi, 1.5*math.pi]
                scenario = 1
            else: # object is above wall
                s_range = [1.5*math.pi, .5*math.pi]
                scenario = 2
        elif object_rect.centery > wall_rect.top and object_rect.centery < wall_rect.bottom:
            if object_rect.right < wall_rect.centerx: #object is to the left of wall
                s_range = [0, math.pi]
                scenario = 3
            else: #object is to the right of wall
                s_range = [math.pi, 0]
                scenario = 4
        elif object_rect.centery > wall_rect.centery: # below the wall
            if object_rect.centerx < wall_rect.centerx: # bottom left:
                s_range = [0.5*math.pi, math.pi]
                scenario = 5
            else: # bottom right
                s_range = [1*math.pi, 1.5*math.pi]
                scenario = 6
        else:  # above the wall
            if object_rect.centerx < wall_rect.centerx:  # top left:
                s_range = [0*math.pi, 0.5*math.pi]
                scenario = 7
            else:  # top right
                s_range = [1.5*math.pi,0]
                scenario = 8
        slist = []
        for i in range(12):
            current_angle = sensor_positions[i]
            currentpos = [math.sin(current_angle)*OBJECT_RADIUS + object_rect.centerx, math.cos(current_angle)*OBJECT_RADIUS + object_rect.centery]

            sensor = -1

            if (angle_in_range(s_range, current_angle) and pass_through_wall(scenario, current_angle, currentpos, wall_rect)):
                if current_angle % (.5 * math.pi) != 0:
                    if current_angle < 0.5*math.pi:
                        if scenario == 2:
                            sensor = (wall_rect.top - currentpos[1])/math.cos(current_angle)
                        elif scenario == 3:
                            sensor = (wall_rect.left - currentpos[0])/math.sin(current_angle)
                        else:
                            sensor = max((wall_rect.top - currentpos[1])/math.cos(current_angle),
                                         (wall_rect.left - currentpos[0])/math.sin(current_angle))
                    elif current_angle < 1*math.pi:
                        if scenario == 3:
                            sensor = (wall_rect.left - currentpos[0])/math.cos(current_angle - 0.5*math.pi)
                        elif scenario == 1:
                            sensor = (currentpos[1] - wall_rect.bottom)/math.sin(current_angle - 0.5*math.pi)
                        else:
                            sensor = max((wall_rect.left - currentpos[0])/math.cos(current_angle - 0.5*math.pi),
                                         (currentpos[1] - wall_rect.bottom)/math.sin(current_angle - 0.5*math.pi))
                    elif current_angle < 1.5*math.pi:
                        if scenario == 1:
                            sensor = (currentpos[1] - wall_rect.bottom) / math.cos(current_angle - 1 * math.pi)
                        elif scenario == 4:
                            sensor = (currentpos[0] - wall_rect.right) / math.sin(current_angle - 1 * math.pi)
                        else:
                            sensor = max((currentpos[1] - wall_rect.bottom) / math.cos(current_angle - 1 * math.pi),
                                         (currentpos[0] - wall_rect.right) / math.sin(current_angle - 1 * math.pi))
                    else:
                        if scenario == 4:
                            sensor = (currentpos[0] - wall_rect.right) / math.cos(current_angle - 1.5 * math.pi)
                        elif scenario == 2:
                            sensor = (wall_rect.top - currentpos[1]) / math.sin(current_angle - 1.5 * math.pi)
                        else:
                            sensor = max((currentpos[0] - wall_rect.right) / math.cos(current_angle - 1.5 * math.pi),
                                         (wall_rect.top - currentpos[1]) / math.sin(current_angle - 1.5 * math.pi))
                else:
                    if current_angle == 0:
                        sensor = wall_rect.top - object_rect.bottom
                    elif current_angle == 0.5 * math.pi:
                        sensor = wall_rect.left - object_rect.right
                    elif current_angle == 1 * math.pi:
                        sensor = object_rect.top - wall_rect.bottom
                    else:
                        sensor = object_rect.left - wall_rect.right
            else:
                if current_angle % (.5*math.pi) != 0:
                    if current_angle < 0.5*math.pi:
                        sensor = min((SCREEN_HEIGHT-currentpos[1])/math.cos(current_angle), (SCREEN_WIDTH-currentpos[0])/math.sin(current_angle))
                    elif current_angle < 1*math.pi:
                        sensor = min((SCREEN_WIDTH-currentpos[0])/math.cos(current_angle - 0.5*math.pi), currentpos[1]/math.sin(current_angle- 0.5*math.pi))
                    elif current_angle < 1.5*math.pi:
                        sensor = min(currentpos[1]/math.cos(current_angle - math.pi), (currentpos[0])/math.sin(current_angle - math.pi))
                    else:
                        sensor = min((currentpos[0])/math.cos(current_angle - 1.5*math.pi), (SCREEN_HEIGHT-currentpos[1])/math.sin(current_angle - 1.5*math.pi))
                else:
                    if current_angle == 0:
                        sensor = SCREEN_HEIGHT - currentpos[1]
                    elif current_angle == 0.5*math.pi:
                        sensor = SCREEN_WIDTH - currentpos[0]
                    elif current_angle == 1*math.pi:
                        sensor = currentpos[1]
                    else:
                        sensor = currentpos[0]


            sensor = min(sensor, 200)
            sensor = round(sensor)
            slist.append(sensor)
        AllSensorVal.append(slist)

    actualVal = []
    for i in range(12):
        actualVal.append(1000)
    for j in range(len(AllSensorVal)):
        #print(AllSensorVal)
        sensors = AllSensorVal[j]
        for i in range(12):
            # print(i)
            # print(actualVal[i])
            # print(sensors)
            actualVal[i] = min(actualVal[i], sensors[i])

    return actualVal

def angle_in_range(range, angle):
    if range[0] < range[1]:
        return range[0] < angle < range[1]
    else:
        return (angle > range[0] or angle < range[1])

def pass_through_wall(scenario, current_angle, currentpos, wall_rect):
    if current_angle % (.5 * math.pi) != 0:
        if current_angle < 0.5 * math.pi:
            if scenario == 2:
                corner = [wall_rect.right, wall_rect.top]
                dis = math.sqrt((currentpos[0]-corner[0])**2 + (currentpos[1]-corner[1])**2)
                angle = math.cos((wall_rect.top-currentpos[1])/dis)
                return current_angle < angle
            elif scenario == 3:
                corner = [wall_rect.left, wall_rect.bottom]
                dis = math.sqrt((currentpos[0] - corner[0]) ** 2 + (currentpos[1] - corner[1]) ** 2)
                angle = math.sin((wall_rect.left - currentpos[0]) / dis)
                return current_angle > angle
            else:
                corner4 = [wall_rect.left, wall_rect.bottom]
                dis = math.sqrt((currentpos[0] - corner4[0]) ** 2 + (currentpos[1] - corner4[1]) ** 2)
                angle4 = math.sin((wall_rect.left - currentpos[0]) / dis)

                corner2 = [wall_rect.right, wall_rect.top]
                dis = math.sqrt((currentpos[0] - corner2[0]) ** 2 + (currentpos[1] - corner2[1]) ** 2)
                angle2 = math.cos((wall_rect.top - currentpos[1]) / dis)

                return angle4 < current_angle < angle2
        elif current_angle < 1 * math.pi:
            if scenario == 3:
                corner = [wall_rect.left, wall_rect.top]
                dis = math.sqrt((currentpos[0] - corner[0]) ** 2 + (currentpos[1] - corner[1]) ** 2)
                angle = math.cos((wall_rect.left - currentpos[0]) / dis)
                return (current_angle - 0.5*math.pi) < angle
            elif scenario == 1:
                corner = [wall_rect.right, wall_rect.bottom]
                dis = math.sqrt((currentpos[0] - corner[0]) ** 2 + (currentpos[1] - corner[1]) ** 2)
                angle = math.sin((currentpos[1] - wall_rect.bottom)/dis)
                return (current_angle - 0.5*math.pi) > angle
            else:
                corner1 = [wall_rect.left, wall_rect.top]
                dis = math.sqrt((currentpos[0] - corner1[0]) ** 2 + (currentpos[1] - corner1[1]) ** 2)
                angle1 = math.cos((wall_rect.left - currentpos[0]) / dis)

                corner3 = [wall_rect.right, wall_rect.bottom]
                dis = math.sqrt((currentpos[0] - corner3[0]) ** 2 + (currentpos[1] - corner3[1]) ** 2)
                angle3 = math.sin((currentpos[1] - wall_rect.bottom) / dis)

                return angle3 < (current_angle - 0.5*math.pi) < angle1
        elif current_angle < 1.5 * math.pi:
            if scenario == 1:
                corner = [wall_rect.left, wall_rect.bottom]
                dis = math.sqrt((currentpos[0] - corner[0]) ** 2 + (currentpos[1] - corner[1]) ** 2)
                angle = math.cos((currentpos[0] - wall_rect.right) / dis)
                return (current_angle - 1*math.pi) < angle

            elif scenario == 4:
                corner = [wall_rect.right, wall_rect.top]
                dis = math.sqrt((currentpos[0] - corner[0]) ** 2 + (currentpos[1] - corner[1]) ** 2)
                angle = math.sin((wall_rect.top - currentpos[1]) / dis)
                return (current_angle - 1 * math.pi) > angle
            else:
                corner4 = [wall_rect.left, wall_rect.bottom]
                dis = math.sqrt((currentpos[0] - corner4[0]) ** 2 + (currentpos[1] - corner4[1]) ** 2)
                angle4 = math.cos((currentpos[0] - wall_rect.right) / dis)

                corner2 = [wall_rect.right, wall_rect.top]
                dis = math.sqrt((currentpos[0] - corner2[0]) ** 2 + (currentpos[1] - corner2[1]) ** 2)
                angle2 = math.sin((wall_rect.top - currentpos[1]) / dis)

                return angle2 < (current_angle - 1 * math.pi) < angle4
        else:
            if scenario == 4:
                corner = [wall_rect.right, wall_rect.bottom]
                dis = math.sqrt((currentpos[0] - corner[0]) ** 2 + (currentpos[1] - corner[1]) ** 2)
                angle = math.cos((currentpos[0] - wall_rect.right) / dis)
                return (current_angle - 1.5*math.pi) < angle

            elif scenario == 2:
                corner = [wall_rect.left, wall_rect.top]
                dis = math.sqrt((currentpos[0] - corner[0]) ** 2 + (currentpos[1] - corner[1]) ** 2)
                angle = math.sin((wall_rect.top-currentpos[1]) / dis)
                return (current_angle - 1.5 * math.pi) > angle
            else:
                corner1 = [wall_rect.left, wall_rect.top]
                dis = math.sqrt((currentpos[0] - corner1[0]) ** 2 + (currentpos[1] - corner1[1]) ** 2)
                angle1 = math.sin((wall_rect.left - currentpos[0]) / dis)

                corner3 = [wall_rect.right, wall_rect.bottom]
                dis = math.sqrt((currentpos[0] - corner3[0]) ** 2 + (currentpos[1] - corner3[1]) ** 2)
                angle3 = math.cos((currentpos[0] - wall_rect.right) / dis)

                return angle1 < (current_angle - 1.5 * math.pi) < angle3

    return True

def calc_number_pos(rotation):

    sensor_positions = []
    locations = []
    for i in range(12):
        value = ((2 * i / 12 * math.pi) + rotation)
        sensor_positions.append(value % (2 * math.pi))
        currentpos = [math.sin(sensor_positions[i]) * (OBJECT_RADIUS + 15) + object_rect.centerx,
                      math.cos(sensor_positions[i]) * (OBJECT_RADIUS + 20) + object_rect.centery]
        locations.append(currentpos)

    return locations


###############################################
############ REMCO ############################
###############################################

def move_robot(left_speed, right_speed):
    # Calculate the average speed of the wheels
    avg_speed = (left_speed + right_speed) / 2.0

    # Calculate the change in direction of the robot based on the difference in speed of the wheels
    direction_change = (right_speed - left_speed) / 100.0

    # Update the direction of the robot
    global robot_direction
    robot_direction += direction_change
    robot_direction %= (2 * math.pi)

    # Calculate the x and y components of the robot's velocity
    velocity_x = avg_speed * math.sin(robot_direction)
    velocity_y = avg_speed * math.cos(robot_direction)

    if velocity_x < -0.25 and velocity_x != 0:
        velocity_x = min(-1, velocity_x)
    elif velocity_x > 0.25 and velocity_x != 0:
        velocity_x = max(1, velocity_x)

    if velocity_y < -0.25 and velocity_y != 0:
        velocity_y = min(-1, velocity_y)
    elif velocity_y > 0.25 and velocity_y != 0:
        velocity_y = max(1, velocity_y)


    # Backwards collision is handled here
    # Calculate next position to see if it goes through wall
    future_object = object_rect.move(velocity_x, velocity_y)

    # see if there is a wall between current object and future object
    # if (is_wall_between(object_rect, future_object, wall_list)):
    #     # Calculate the nearest point on the wall to the robot's current position
    #     nearest_point = nearest_point_on_wall(object_rect, wall_rect, velocity_x, velocity_y)
    #     # Set the robot's position to the nearest point on the wall
    #     object_rect.centerx = nearest_point[0]
    #     # object_rect.center = nearest_point
    #     # Stop further checking for collisions
    #     return
    # Update the position of the robot
    object_rect.move_ip(velocity_x, velocity_y)


def is_wall_between(object_rect1, object_rect2, walls):
    # Calculate the x and y distance between the two objects
    x_dist = object_rect2.centerx - object_rect1.centerx
    y_dist = object_rect2.centery - object_rect1.centery

    # Calculate the step size for x and y distances
    x_step = x_dist / abs(x_dist) if x_dist != 0 else 0
    y_step = y_dist / abs(y_dist) if y_dist != 0 else 0

    # Initialize the starting position and the number of steps
    x, y = object_rect1.centerx, object_rect1.centery
    num_steps = max(abs(x_dist), abs(y_dist))

    # Check for walls along the path between the two objects
    for i in range(num_steps):
        # Check if the current position is inside any of the walls
        for wall in walls:
            if wall.collidepoint(x, y):
                return True

        # Update the position
        x += x_step
        y += y_step

    # If no walls were found, return False
    return False
def nearest_point_on_wall(object_rect, wall_rect, velocity_x, velocity_y):
    # Find the closest point on the wall to the current position of the object
    closest_x = max(wall_rect.left, min(object_rect.centerx + velocity_x, wall_rect.right))
    closest_y = max(wall_rect.top, min(object_rect.centery + velocity_y, wall_rect.bottom))

    # Determine if the object is moving towards or away from the wall
    moving_towards_wall = ((object_rect.centerx - closest_x) * velocity_x <= 0) or ((object_rect.centery - closest_y) * velocity_y <= 0)

    # If the object is moving towards the wall, return the closest point on the wall
    if moving_towards_wall:
        return closest_x, closest_y

    # Otherwise, return the current position of the object
    return object_rect.centerx, object_rect.centery


# JAN
def add_coordinates_within_radius(center, radius, coords_set):
    x_start = center[0] - radius
    x_end = center[0] + radius + 1
    y_start = center[1] - radius
    y_end = center[1] + radius + 1

    # Create arrays of x and y coordinates
    x_coords = np.arange(x_start, x_end)
    y_coords = np.arange(y_start, y_end)

    # Compute the squared distances from each pixel to the center point
    x_diffs = x_coords - center[0]
    y_diffs = y_coords - center[1]
    dists_squared = x_diffs[:, np.newaxis] ** 2 + y_diffs ** 2

    # Create a mask of pixels within the given radius of the center point
    mask = dists_squared <= radius ** 2

    # Extract the coordinates of the pixels that satisfy the mask
    coords = np.argwhere(mask) + np.array([[y_start, x_start]])

    # Add the coordinates to the input set
    coords_set.update(set(map(tuple, coords)))







# JEAN & REMCO
def run_game(weights=None, max_timesteps = 5000, scr = True, other_fitness=False):
    # Initialize Pygame
    pygame.init()

    # Define constants for the screen size
    global SCREEN_WIDTH
    global SCREEN_HEIGHT
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 1000

    # Create the screen and set its caption
    if scr:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Object Movement")

    # Define constants for the object size and movement speed
    global OBJECT_WIDTH
    OBJECT_WIDTH = 50
    global OBJECT_HEIGHT
    OBJECT_HEIGHT = 50
    MOVEMENT_SPEED = 1
    INCREMENT = 1
    global OBJECT_RADIUS

    OBJECT_RADIUS = 25

    # variables for arrows
    ARROW_LENGTH = 50
    ARROW_WIDTH = 20

    # Speed for both wheels
    left_speed = 0
    right_speed = 0

    # direction
    global robot_direction
    robot_direction = math.pi / 2

    # Create the object and set its initial position
    global object_rect
    object_rect = pygame.Rect(0, 0, OBJECT_WIDTH, OBJECT_HEIGHT)
    # object_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    # FOR SIMPLE MAP
    # object_rect.center = (570, SCREEN_HEIGHT // 2)
    # FOR HARDER MAP
    object_rect.center = (150, 175)


    # Define constants for the wall size and position
    global WALL_WIDTH
    global WALL_HEIGHT
    global WALL_X
    global WALL_Y
    WALL_WIDTH = 200
    WALL_HEIGHT = SCREEN_HEIGHT // 2
    WALL_X = SCREEN_WIDTH // 3
    WALL_Y = SCREEN_HEIGHT // 4

    # Create the wall rect and set its position
    global wall_rect
    # wall_rect = pygame.Rect(WALL_X, WALL_Y, WALL_WIDTH, WALL_HEIGHT)
    wall2 = pygame.Rect(50, 50, 50, 200)
    wall3 = pygame.Rect(50, 50, 900, 50)
    wall4 = pygame.Rect(50, 250, 600, 50)
    wall5 = pygame.Rect(950, 50, 50, 800)
    wall6 = pygame.Rect(650, 250, 50, 600)
    # Create a font object to display the number
    if scr:
        font = pygame.font.Font(None, 24)
    object_speed = [0, 0]

    L = 0.2  # distance between the wheels
    r = 0.05  # radius of the wheels
    DELTA_T = 0.01  # time interval for each iteration of the control loop

    # Create a set to track the pressed keys
    pressed_keys = set()
    second_pressed = set()

    wall_list = [wall2, wall3, wall4, wall5, wall6]

    # Set up the set to keep track of visited coordinates
    visited = set()
    True_Visited = set()

    # Create a game loop
    running = True

    timestep = 0

    # initialize NN according to given weights
    if weights!=None:
        nn = ANN(weights1=weights[0], weights2=weights[1])

    punishment = 0
    while running:
        # Moved here because needed in ANN
        distanceList = circle_sensor_calc(robot_direction, wall_list)

        if timestep > max_timesteps:
            pygame.quit()
            # Return fitness here (just for test the amount of area cleaned)
            if other_fitness:
                return 200*len(visited) - punishment
                # return - punishment
            else:
                return len(True_Visited)
        timestep += 1
        # Handle events
        if weights != None:


            # Left speed and right speed outputs of NN
            left_speed, right_speed = nn.predict(distanceList)

            punishment += 2 * abs(left_speed - right_speed)

            punishment += - 2 * abs(left_speed) - 2 * abs(right_speed)

            for distance in distanceList:
                if distance < 10:
                    punishment += 100/(abs(distance) + 1)

            # for testing, divide by 100 for normalization
            # left_speed = left_speed/100
            # right_speed = right_speed/100

            left_speed = round(left_speed)
            right_speed = round(right_speed)

            # limit speed between -10 and 10
            # MAX_SPEED = 50
            # left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
            # right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    ###############################################
                    ############ REMCO ############################
                    ###############################################
                    # Add the pressed key to the set
                    pressed_keys.add(event.key)

                    if event.key == pygame.K_w:
                        left_speed += INCREMENT
                    elif event.key == pygame.K_s:
                        left_speed -= INCREMENT
                    elif event.key == pygame.K_o:
                        right_speed += INCREMENT
                    elif event.key == pygame.K_l:
                        right_speed -= INCREMENT
                    elif event.key == pygame.K_x:
                        right_speed = 0
                        left_speed = 0
                    elif event.key == pygame.K_t:
                        left_speed += INCREMENT
                        right_speed += INCREMENT
                    elif event.key == pygame.K_g:
                        left_speed -= INCREMENT
                        right_speed -= INCREMENT
                    # Extra keys for testing high speeds
                    elif event.key == pygame.K_n:
                        left_speed -= INCREMENT*1000
                        right_speed -= INCREMENT*1000
                    elif event.key == pygame.K_m:
                        left_speed += INCREMENT*1000
                        right_speed += INCREMENT*1000

                elif event.type == pygame.KEYUP:
                    # Remove the released key from the set
                    pressed_keys.discard(event.key)
        move_robot(left_speed, right_speed)



        # Handle collision with walls
        for wall in wall_list:
            if object_rect.colliderect(wall):
                handle_collision(wall)

        handle_outside_collision()

        # Clear the screen
        if scr:
            screen.fill((255, 255, 255))

        # Add the current coordinate to the set
        True_Visited.add(object_rect.center)
        rounded = tuple(round(ti/25) for ti in object_rect.center)
        #visited.add(rounded)

        # count already visited coordinates for penalty
        if rounded in visited:
            punishment += 10
            #punishment += 0

        else:
            visited.add(rounded)
        # Calculate the area covered
        area = len(visited)

        if scr:
            for coord in True_Visited:
                pygame.draw.circle(screen, (210,105,30), coord, OBJECT_RADIUS)

        # r = 1


            pygame.draw.circle(screen, (255, 0, 0), object_rect.center, OBJECT_RADIUS)
            ###############################################
            ############ JEAN ############################
            ###############################################
            # Calculate the position of the left and right wheels
            left_wheel_pos = (object_rect.centerx - (OBJECT_WIDTH / 2) * math.sin(robot_direction - math.pi / 2),
                              object_rect.centery - (OBJECT_WIDTH / 2) * math.cos(robot_direction - math.pi / 2))
            right_wheel_pos = (object_rect.centerx + (OBJECT_WIDTH / 2) * math.sin(robot_direction - math.pi / 2),
                               object_rect.centery + (OBJECT_WIDTH / 2) * math.cos(robot_direction - math.pi / 2))

            # Draw the left and right wheels as circles
            pygame.draw.circle(screen, (0, 255, 0), left_wheel_pos, OBJECT_RADIUS)
            pygame.draw.circle(screen, (0, 255, 0), right_wheel_pos, OBJECT_RADIUS)

            # Draw a line between the left and right wheels
            pygame.draw.line(screen, (0, 0, 255), left_wheel_pos, right_wheel_pos, 10)

            for wall in wall_list:
                pygame.draw.rect(screen, (0, 0, 255), wall)

            end_point = (object_rect.centerx + ARROW_LENGTH * math.sin(robot_direction),
                         object_rect.centery + ARROW_LENGTH * math.cos(robot_direction))
            pygame.draw.line(screen, (0, 0, 255), object_rect.center, end_point, 5)

            #distanceList = sensor_calc()
            #TODO enter list walls here
            #distanceList = circle_sensor_calc(0)

            # Render the number as text
            number_text = font.render('{}'.format(distanceList[0]), True, (0, 0, 0))

            # Get the dimensions of the text
            text_width, text_height = font.size('1')

            locations = calc_number_pos(robot_direction)
            #locations = calc_number_pos(0)
            ###############################################
            ############ JAN ############################
            ###############################################
            # Draw the text next to the object
            screen.blit(font.render('{}'.format(distanceList[0]), True, (0, 0, 0)), (locations[0][0], locations[0][1]))
            screen.blit(font.render('{}'.format(distanceList[1]), True, (0, 0, 0)), (locations[1][0], locations[1][1]))
            screen.blit(font.render('{}'.format(distanceList[2]), True, (0, 0, 0)), (locations[2][0], locations[2][1]))
            screen.blit(font.render('{}'.format(distanceList[3]), True, (0, 0, 0)), (locations[3][0], locations[3][1]))
            screen.blit(font.render('{}'.format(distanceList[4]), True, (0, 0, 0)), (locations[4][0], locations[4][1]))
            screen.blit(font.render('{}'.format(distanceList[5]), True, (0, 0, 0)), (locations[5][0], locations[5][1]))
            screen.blit(font.render('{}'.format(distanceList[6]), True, (0, 0, 0)), (locations[6][0], locations[6][1]))
            screen.blit(font.render('{}'.format(distanceList[7]), True, (0, 0, 0)), (locations[7][0], locations[7][1]))
            screen.blit(font.render('{}'.format(distanceList[8]), True, (0, 0, 0)), (locations[8][0], locations[8][1]))
            screen.blit(font.render('{}'.format(distanceList[9]), True, (0, 0, 0)), (locations[9][0], locations[9][1]))
            screen.blit(font.render('{}'.format(distanceList[10]), True, (0, 0, 0)), (locations[10][0], locations[10][1]))
            screen.blit(font.render('{}'.format(distanceList[11]), True, (0, 0, 0)), (locations[11][0], locations[11][1]))

            # Draw the area covered
            font = pygame.font.Font(None, 36)
            text = font.render("Area covered: {} pixels".format(area), True, (0, 0, 0))
            screen.blit(text, (0, 20))

            font1 = pygame.font.Font(None, 40)
            screen.blit(font1.render('left speed: {}, right speed: {}'.format(left_speed, right_speed), True, (0, 0, 0)), (0, 0))

            # Update the screen
            pygame.display.flip()


    # Quit Pygame
    pygame.quit()
# # np.random.seed(0)

# nn = ANN(weights1=W1, weights2=W2)
#
# # Test the neural network
# # exit(0)
# prev_val = 0
# for i in range(100):
#     print(i)
#     fitness = run_game(weights=[W1, W2], scr=False)
#
#     if prev_val != fitness:
#         print("test")
#         print(fitness)
#         print(prev_val)
#
#     prev_val = fitness
# # print(run_game(weights=[W1, W2]))
# print(run_game(weights=[W1, W2]))
# print(run_game(weights=[W1, W2]))
# print(run_game(weights=[W1, W2]))
# print(run_game(weights=[W1, W2]))
