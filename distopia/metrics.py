"""
States Metrics
=================

Defines metrics that summarize the state based on the districts.
"""
__all__ = ('StateMetric', )


class StateMetric(object):
    """Metrics used across :class:`distopia.district.District`.
    """

    name = ''
    """A globally unique metric name that describes the metric.
    """

    districts = []

    def __init__(self, name, districts, **kwargs):
        super(StateMetric, self).__init__(**kwargs)
        self.name = name
        self.districts = districts

    def compute(self):
        raise NotImplementedError
