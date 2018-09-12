"""
District Metrics
=================

Defines metrics that summarize the district based on the contained precincts
and other metadata.
"""

__all__ = ('DistrictMetric', )


class DistrictMetric(object):
    """Metrics used with :class:`distopia.district.District`.
    """

    name = ''
    """A globally unique metric name that describes the metric.
    """
