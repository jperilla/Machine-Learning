class Car:

    def __init__(self, x, y, vx, vy, max_x, max_y):
        self.position = CarPosition(x, y)
        self.last_position = CarPosition(x, y)
        self.velocity = CarVelocity(vx, vy)
        self.last_velocity = CarVelocity(vx, vy)
        self.max_x = max_x
        self.max_y = max_y

    def accelerate(self, ax, ay):
        self.last_position = CarPosition(self.position.x, self.position.y)
        self.last_velocity = CarVelocity(self.velocity.x, self.velocity.y)
        self.velocity.accelerate(ax, ay)
        self.__move()

    def set_to_last_position(self):
        self.position = self.last_position

    def reset_velocity(self):
        self.velocity.x = 0
        self.velocity.y = 0

    def get_state(self):
        return tuple((self.position.x, self.position.y, self.velocity.x, self.velocity.y))

    def set_state(self, state):
        self.position.x = state[0]
        self.position.y = state[1]
        self.velocity.x = state[2]
        self.velocity.y = state[3]

    def __move(self):
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y

        # Limit to max values on track
        if self.position.x < 0:
            self.position.x = 0

        if self.position.x >= self.max_x:
            self.position.x = self.max_x - 1

        if self.position.y < 0:
            self.position.y = 0

        if self.position.y >= self.max_y:
            self.position.y = self.max_y - 1


class CarPosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class CarVelocity:
    def __init__(self, vx, vy):
        self.x = vx
        self.y = vy

    def accelerate(self, ax, ay):
        if self.x + ax > 5:
            self.x = 5
        elif self.x + ax < -5:
            self.x = -5
        else:
            self.x += ax

        if self.y + ay > 5:
            self.y = 5
        elif self.y + ay < -5:
            self.y = -5
        else:
            self.y += ay



