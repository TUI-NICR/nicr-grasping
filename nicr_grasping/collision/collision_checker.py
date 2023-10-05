class CollisionChecker:
    def __init__(self):
        pass

    def check_collision(self, grasp):
        raise NotImplementedError

    @property
    def collision_info(self):
        raise NotImplementedError
