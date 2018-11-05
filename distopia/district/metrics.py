"""
District Metrics
=================

Defines metrics that summarize the district based on the contained precincts
and other metadata.
"""
import numpy as np

__all__ = ('DistrictMetric', 'DistrictAggregateMetric',
           'DistrictHistogramAggregateMetric')


class DistrictMetric(object):
    """Metrics used with :class:`distopia.district.District`.
    """

    name = ''
    """A globally unique metric name that describes the metric.
    """

    def __init__(self, name, **kwargs):
        super(DistrictMetric, self).__init__(**kwargs)
        self.name = name

    def compute(self):
        raise NotImplementedError


class DistrictAggregateMetric(DistrictMetric):

    district = None

    scalar_value = 0

    scalar_maximum = 0

    scalar_label = ''

    def __init__(self, district, **kwargs):
        super(DistrictAggregateMetric, self).__init__(**kwargs)
        self.district = district

    def compute(self):
        raise NotImplementedError


class DistrictHistogramAggregateMetric(DistrictAggregateMetric):

    data = []

    labels = []

    precinct_metrics = []

    def compute(self):
        name = self.name
        if not self.district.precincts:
            return

        self.precinct_metrics = precinct_metrics = \
            [p.metrics[name] for p in self.district.precincts]
        if not precinct_metrics:
            return

        data = [m.data for m in precinct_metrics]

        self.labels = precinct_metrics[0].labels
        self.data = np.sum(np.array(data), axis=0).tolist()

    def compute_scalar_sum(self):
        precinct_metrics = self.precinct_metrics
        self.scalar_maximum = float(
            np.sum([m.scalar_maximum for m in precinct_metrics]))
        self.scalar_value = float(
            np.sum([m.scalar_value for m in precinct_metrics]))
        self.scalar_label = precinct_metrics[0].scalar_label

    def compute_scalar_mean(self):
        precinct_metrics = self.precinct_metrics
        self.scalar_maximum = float(precinct_metrics[0].scalar_maximum)
        self.scalar_value = float(
            np.mean([m.scalar_value for m in precinct_metrics]))
        self.scalar_label = precinct_metrics[0].scalar_label

    def get_data(self):
        return {
            "name": self.name, "labels": self.labels, "data": self.data,
            "scalar_value": self.scalar_value,
            "scalar_maximum": self.scalar_maximum,
            "scalar_label": self.scalar_label}
