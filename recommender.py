class Recommender:

    DEFAULT_NUM_FEATURES = 10
    DEFAULT_REGULARIZATION = 1

    def __init__(self, num_features=DEFAULT_NUM_FEATURES,
                 reg=DEFAULT_REGULARIZATION, Y=None, R=None):
        self.num_features = num_features
        self.reg = reg
        self.Y = Y
        self.R = R
        self.Theta = None
        self.X = None

        if (Y is not None) and (R is not None):
            self.learn(Y, R)

    def learn(self, Y, R, reg=None):
        if reg is not None:
            self.reg = reg
