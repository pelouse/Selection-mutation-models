from measures.measures import Measure


# A faster way to create the identity measure
class Id(Measure):

    def __init__(self, interval=[0, 1]):
        super().__init__(lambda x: x, interval=interval)


# A faster way to create the uniform measure
class OneMeasure(Measure):

    def __init__(self, interval=[0, 1]):
        super().__init__(
            lambda x: 1 + 0 * x * (interval[0] <= x) * (x <= interval[1]),
            interval=interval,
        )
