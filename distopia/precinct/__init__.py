"""
Precinct
=========

:class:`Precinct` defines a precinct and it's immutable data.
"""

from collections import deque, defaultdict

__all__ = ('Precinct', )


class Precinct(object):
    """
    Describes a precinct and its data.
    """

    name = ''
    """Name describing the precinct.
    """

    identity = 0
    """The id of the precinct. """

    boundary = []
    """A nx2 list of the ``x``, ``y`` coordinates of the polygon that
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

    collider = None

    def __init__(self, boundary=None, identity=0, name='', location=(0, 0),
                 **kwargs):
        super(Precinct, self).__init__(**kwargs)
        if boundary is None:
            boundary = []
        self.identity = identity
        self.location = location
        self.boundary = boundary
        self.name = name
        self.neighbours = []
        self.metrics = {}

    @classmethod
    def find_disconnected_precincts(cls, precincts, district_map):
        seen = {precincts[0]}
        stack = deque([precincts[0]])
        district = district_map[precincts[0]]

        for precinct in precincts:
            assert district_map[precinct] is district

        while stack:
            p = stack.popleft()

            for neighbor in p.neighbours:
                if neighbor not in seen and district_map[neighbor] is district:
                    stack.appendleft(neighbor)
                    seen.add(neighbor)

        unseen = [p for p in precincts if p not in seen]
        if unseen:
            unseen.append(precincts[0])
        return unseen
