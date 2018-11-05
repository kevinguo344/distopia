"""
Precinct Metrics
=================

Defines metrics that are specific to Precincts.
"""

__all__ = ('PrecinctMetric', 'PrecinctHistogram')


class PrecinctMetric(object):
    """Metrics used with :class:`distopia.precinct.Precinct`.
    """

    name = ''
    """A globally unique metric name that describes the metric.
    """

    scalar_value = 0

    scalar_maximum = 0

    scalar_label = ''

    def __init__(
            self, name, scalar_value=0, scalar_maximum=0, scalar_label='',
            **kwargs):
        super(PrecinctMetric, self).__init__(**kwargs)
        self.name = name
        self.scalar_value = scalar_value
        self.scalar_maximum = scalar_maximum
        self.scalar_label = scalar_label


class PrecinctHistogram(PrecinctMetric):

    data = []

    labels = []

    def __init__(self, data, labels, **kwargs):
        super(PrecinctHistogram, self).__init__(**kwargs)
        self.data = data
        self.labels = labels

    def get_data(self):
        return {
            "name": self.name, "labels": self.labels, "data": self.data,
            "scalar_value": self.scalar_value,
            "scalar_maximum": self.scalar_maximum,
            "scalar_label": self.scalar_label}
