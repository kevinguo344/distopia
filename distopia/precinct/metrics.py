"""
Precinct Metrics
=================

Defines metrics that are specific to Precincts.
"""

__all__ = ('PrecinctMetric', )


class PrecinctMetric(object):
    """Metrics used with :class:`distopia.precinct.Precinct`.
    """

    name = ''
    """A globally unique metric name that describes the metric.
    """
