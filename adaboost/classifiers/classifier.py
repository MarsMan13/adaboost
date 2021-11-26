import abc


class Classifier:
    """MARSMAN13
        Classifier is a stump for any feature
        
    """
    def __init__(self, feature):
        self.error = None
        self.precision = None
        self.feature = feature  # type: Simple

    @abc.abstractmethod
    def describe(self):
        return ''

    @abc.abstractmethod
    def ready_data(self, data, actual, weights):
        """MARSMAN13
            
            <params>
            * data    : data matrix, shape - "no.entities x no.features"
            * actual  : A column vector of actual values (results) (-1 or 1 value)
            * weights : A column vector of weights of each entity
            
            <process>
            fills error, precision parameters based on the data
        
        """
        """
        Takes a data matrix, a column vector of actual classifications (-1 or 1) and a column vector of weights and
        customizes the classifier accordingly (e.g., sets the threshold). If this function returns true, it must
        fill in the error and precision parameters based on the data.
        :param data: np.ndarray
        :param actual: np.ndarray
        :param weights: np.ndarray
        :return: bool
        """
        pass

    @abc.abstractmethod
    def classify_data(self, data):
        """MARSMAN13
            predicts results (-1 or 1)
            
        """
        
        """
        Takes a data matrix and returns a set of predicted classifications (either -1 or 1).
        :param data: np.ndarray
        :return: np.ndarray
        """
        pass
