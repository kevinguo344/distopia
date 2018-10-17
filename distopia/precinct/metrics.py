"""
Precinct Metrics
=================

Defines metrics that are specific to Precincts.
"""

__all__ = ('PrecinctMetric', 'PrecinctHistogram', 'PrecinctScalar',
           'PrecinctCategory')


class PrecinctMetric(object):
    """Metrics used with :class:`distopia.precinct.Precinct`.
    """

    name = ''
    """A globally unique metric name that describes the metric.
    """

    def __init__(self, name, **kwargs):
        super(PrecinctMetric, self).__init__(**kwargs)
        self.name = name


class PrecinctHistogram(PrecinctMetric):

    data = []

    labels = []

    def __init__(self, data, labels, **kwargs):
        super(PrecinctHistogram, self).__init__(**kwargs)
        self.data = data
        self.labels = labels


class PrecinctScalar(PrecinctMetric):

    value = 0

    def __init__(self, value, **kwargs):
        super(PrecinctScalar, self).__init__(**kwargs)
        self.value = value


class PrecinctCategory(PrecinctMetric):

    value = 0

    def __init__(self, value, **kwargs):
        super(PrecinctCategory, self).__init__(**kwargs)
        self.value = value
