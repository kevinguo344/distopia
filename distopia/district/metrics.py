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

    def __init__(self, district, **kwargs):
        super(DistrictAggregateMetric, self).__init__(**kwargs)
        self.district = district

    def compute(self):
        raise NotImplementedError


class DistrictHistogramAggregateMetric(DistrictAggregateMetric):

    data = []

    labels = []

    def compute(self):
        name = self.name
        if not self.district.precincts:
            return

        self.labels = self.district.precincts[0].metrics[name].labels
        precinct_metrics = [
            p.metrics[name].data for p in self.district.precincts]

        self.data = np.sum(np.array(precinct_metrics), axis=0).tolist()
