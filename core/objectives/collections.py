class LossCollection:

    def __init__(self, total, **observables):
        self.total = total
        self.observables = observables

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        if isinstance(other, LossCollection):
            return LossCollection(
                self.total + other.total,
                **self.observables,
                **other.observables,
            )
        elif other == 0:
            return LossCollection(self.total + 0, **self.observables)
        else:
            raise TypeError
