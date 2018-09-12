"""
District
=========

:class:`District` defines a district and and the precincts it contains.
"""

__all__ = ('District', )


class District(object):
    """
    Describes a district, its precincts and its metrics.
    """

    name = ''
    """A globally unique name (or number) describing the district.
    """

    boundary = []
    """A list of the ``x``, ``y`` coordinates of the polygon that 
    describes the district's boundary.
    """

    neighbours = []
    """List of other :class:`District`'s that are on the boundary of this
    district.
    """

    precincts = []
    """List of :class:`~distopia.precinct.Precinct` instances that are 
    currently contained within this district.
    """

    metrics = {}
    """A mapping from :attr:`~distopia.district.metrics.DistrictMetric.name` 
    to the :class:`~distopia.district.metrics.DistrictMetric` instance that
    contains the summery metric data for this district.
    """

    def __init__(self, **kwargs):
        super(District, self).__init__(**kwargs)
        self.boundary = []
        self.neighbours = []
        self.precincts = []
        self.metrics = {}
