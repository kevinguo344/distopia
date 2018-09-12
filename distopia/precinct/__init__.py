"""
Precinct
=========

:class:`Precinct` defines a precinct and it's immutable data.
"""

__all__ = ('Precinct', )


class Precinct(object):
    """
    Describes a precinct and its data.
    """

    name = ''
    """A globally unique name (or number) describing the precinct.
    """

    boundary = []
    """A list of the ``x``, ``y`` coordinates of the polygon that 
    describes the precinct's boundary.
    """

    location = (0, 0)
    """The (practically unique) center location of the precinct.
    """

    neighbours = []
    """List of other :class:`Precinct`'s that are on the boundary of this
    precinct.
    """

    district = None
    """The :class:`~distopia.district.District` that currently contains this
    precinct.
    """

    metrics = {}
    """A mapping from :attr:`~distopia.precinct.metrics.PrecinctMetric.name` 
    to the :class:`~distopia.precinct.metrics.PrecinctMetric` instance that
    contains the metric data for this precinct.
    """

    def __init__(self, **kwargs):
        super(Precinct, self).__init__(**kwargs)
        self.boundary = []
        self.neighbours = []
        self.metrics = {}
