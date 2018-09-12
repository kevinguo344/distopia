"""
Voroni Mapping
===============
"""
from scipy.spatial import Voronoi

__all__ = ('VoroniMapping', )


class VoroniMapping(object):
    """Uses the Voroni algorithm to assign precincts to districts.
    """

    sites = []
    """A list of tuples describing the ``x``, ``y`` coordinates of each of the
    objects placed by the user at that location.
    """

    precincts = []
    """A list of all the :class:`distopia.precinct.Precinct` instances.
    """

    districts = []
    """A list of all current :class:`distopia.district.District` instances.
    """

    def __init__(self, **kwargs):
        super(VoroniMapping, self).__init__(**kwargs)
        self.sites = []
        self.precincts = []
        self.districts = []
