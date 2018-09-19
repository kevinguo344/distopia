"""
Voronoi Mapping
===============
"""
from scipy.spatial import Voronoi
from distopia.district import District
from kivy.garden.collider import Collide2DPoly as PolygonCollider
import numpy as np
from collections import defaultdict

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

    precinct_colliders = []
    """A list of :class:`~distopia.utils.PolygonCollider`, each corresponding
    to a precinct in :attr:`precincts`.
    """

    districts = []
    """A list of all current :class:`distopia.district.District` instances.
    """

    district_colliders = []
    """A list of :class:`~distopia.utils.PolygonCollider`, each corresponding
    to a district in :attr:`districts`.
    """

    screen_size = (1920, 1080)
    """The size of the screen, in pixels, on which the districts and
    precincts are drawn, and on which the fiducials are drawn.

    It's ``(width, height)``
    """

    fiducial_locations = []
    """List of tuples, each containing the ``(x, y)`` coordinates of the
    fiducial at that index.
    """

    pixel_precinct_map = None
    """A width by height matrix, where each item is the precinct index in
    :attr:`precincts` it belongs to, or ``2 ** 16 - 1`` if None.
    """

    pixel_district_map = None
    """A width by height matrix, where each item is the district index in
    :attr:`districts` it belongs to.
    """

    def __init__(self, **kwargs):
        super(VoronoiMapping, self).__init__(**kwargs)
        self.sites = []
        self.precincts = []
        self.districts = []
        self.precinct_colliders = []
        self.fiducial_locations = []

    def set_precincts(self, precincts):
        """Adds the precincts to be used by the mapping.

        Must be called only (or every time) after :attr:`screen_size` is set.

        :param precincts: List of :class:`distopia.precinct.Precinct`
            instances.
        """
        w, h = self.screen_size
        pixel_precinct_map = self.pixel_precinct_map = np.ones(
            (w, h), dtype=np.uint16) * (2 ** 16 - 1)

        self.precincts = list(precincts)
        colliders = self.precinct_colliders = [
            PolygonCollider(points=precinct.boundary, cache=True) for
            precinct in precincts]

        for i, (precinct, collider) in enumerate(zip(precincts, colliders)):
            for x, y in collider.get_inside_points():
                if 0 <= x < w and 0 <= y < h:
                    pixel_precinct_map[x, y] = i

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

    def remove_fiducial(self, fiducial):
        """Removes ``fiducial`` from the diagram.

        :param fiducial: ``fiducial`` ID as returned by :meth:`add_fiducial`.
        """
        del self.fiducial_locations[fiducial]

    def get_fiducials(self):
        """Returns a list of the fiducials, indexed by their ID.
        """
        return self.fiducial_locations

    def assign_precincts_to_districts(self):
        """Uses the pre-computed precinct and district maps and assigns
        all the precincts to districts.
        """
        precincts = self.precincts
        districts = self.districts

        precinct_map = self.pixel_precinct_map
        district_map = self.pixel_district_map

        for district in districts:
            district.clear()

        for i, precinct in enumerate(precincts):
            district_indices = district_map[precinct_map == i]
            indices, counts = np.unique(district_indices, return_counts=True)
            district = districts[indices[np.argmax(counts)]]

            district.add_precinct(precinct)

    def compute_district_pixels(self):
        """Computes the assignment of pixels to districts and creates the
        associated districts.
        """
        fiducials = np.asarray(self.fiducial_locations)
        vor = Voronoi(fiducials)
        regions, vertices = self.voronoi_finite_polygons_2d(vor)

        w, h = self.screen_size
        pixel_district_map = self.pixel_district_map = np.ones(
            (w, h), dtype=np.uint16) * (2 ** 16 - 1)

        self.districts = districts = []
        self.district_colliders = colliders = []

        for i, region_indices in enumerate(regions):
            poly = list(map(float, vertices[region_indices].reshape((-1, ))))
            collider = PolygonCollider(points=poly, cache=True)

            district = District()
            district.name = str(i)
            district.boundary = poly

            districts.append(district)
            colliders.append(collider)

            for x, y in collider.get_inside_points():
                if 0 <= x < w and 0 <= y < h:
                    pixel_district_map[x, y] = i

    def voronoi_finite_polygons_2d(self, vor):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        based on https://stackoverflow.com/a/20678647.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """
        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        w, h = self.screen_size
        radius = 100 * max(w, h)

        # Construct a map containing all ridges for a given point
        all_ridges = defaultdict(list)
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges[p1].append((p2, v1, v2))
            all_ridges[p2].append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):  # region of each point
            vertices = vor.regions[region]  # indices of region vertices

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)
