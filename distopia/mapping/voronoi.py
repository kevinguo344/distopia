"""
Voronoi Mapping
===============
"""
from scipy.spatial import Voronoi

__all__ = ('VoronoiMapping', )


class VoronoiMapping(object):
    """Uses the Voronoi algorithm to assign precincts to districts.
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

    screen_size = (1920, 1080)
    """The size of the screen, in pixels, on which the districts and
    precincts are drawn, and on which the fiducials are drawn.
    """

    fiducial_locations = []
    """List of tuples, each containing the ``(x, y)`` coordinates of the
    fiducial at that index.
    """

    def __init__(self, **kwargs):
        super(VoronoiMapping, self).__init__(**kwargs)
        self.sites = []
        self.precincts = []
        self.districts = []
        self.fiducial_locations = []

    def add_fiducial(self, location):
        """Adds a new fiducial at ``location``.

        :param location: The fiducial location ``(x, y)``.
        :return: The ID of the fiducial.
        """
        i = len(self.fiducial_locations)
        self.fiducial_locations.append(location)
        return i

    def move_fiducial(self, fiducial, location):
        """Moves ``fiducial`` from its previous location to a new location.

        :param fiducial: ``fiducial`` ID as returned by :meth:`add_fiducial`.
        :param location: The new fiducial location ``(x, y)``.
        """
        self.fiducial_locations[fiducial] = location
