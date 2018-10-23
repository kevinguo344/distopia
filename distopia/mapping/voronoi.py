"""
Voronoi Mapping
===============
"""
from scipy.spatial import Voronoi
from distopia.district import District
from distopia.mapping._voronoi import PolygonCollider
import numpy as np
from collections import defaultdict
import logging
import time
from threading import Thread, Lock
import math
import cProfile, pstats, io
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

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

    screen_size = (1920, 1080)
    """The size of the screen, in pixels, on which the districts and
    precincts are drawn, and on which the fiducials are drawn.

    It's ``(width, height)``
    """

    fiducial_locations = {}
    """dict of tuples, each containing the ``(x, y)`` coordinates of the
    fiducial at that key.
    """

    fiducial_ids = {}

    pixel_district_map = None
    """A width by height matrix, where each item is the district index in
    :attr:`districts` it belongs to. Read_only (used by the thread).
    """

    precinct_indices = []

    _fiducial_count = 0

    _thread = None

    _thread_queue = None

    _profiler = None

    thread_lock = None

    _raw_districts = []

    def __init__(self, **kwargs):
        super(VoronoiMapping, self).__init__(**kwargs)
        self._profiler = cProfile.Profile()
        self.sites = []
        self.precincts = []
        self.districts = []
        self.precinct_colliders = []
        self.fiducial_locations = {}
        self.fiducial_ids = {}
        self.thread_lock = Lock()
        self._thread_queue = Queue()
        self._thread = thread = Thread(
            target=self.voronoi_thread_function)
        thread.start()

    def post_thread_computation_callback(
            self, districts, raw_districts, precinct_assignment,
            pixel_district_map):
        self.districts = districts
        self._raw_districts = raw_districts
        self.pixel_district_map = pixel_district_map
        for district, precincts in zip(districts, precinct_assignment):
            district.assign_precincts(precincts)

    def voronoi_thread_function(self):
        # this thread never modifies any properties of existing precincts or
        # districts to prevent thread safety issues.
        queue = self._thread_queue
        lock = self.thread_lock
        post_callback = self.post_thread_computation_callback

        while True:
            item = queue.get(block=True)
            if item == 'eof':
                s = io.StringIO()
                try:
                    ps = pstats.Stats(
                        self._profiler, stream=s).sort_stats('cumulative')
                    ps.print_stats()
                    print(s.getvalue())
                except TypeError:  # in case nothing was profiled yet
                    pass
                return

            callback, callback_if_old, fiducials, fiducial_ids = item
            if fiducials is None:
                with lock:
                    fiducials = dict(self.fiducial_locations)
                    fiducial_ids = self.fiducial_ids

            if len(fiducials) <= 3:
                callback([], [], [])
                continue

            fiducial_keys = list(fiducials.keys())
            fiducial_pos = [fiducials[key] for key in fiducial_keys]
            fiducial_identity = [fiducial_ids[key] for key in fiducial_keys]

            self._profiler.enable()
            try:
                districts, raw_districts, pixel_district_map = \
                    self.compute_district_pixels(
                        np.asarray(fiducial_pos), fiducial_identity)
                if not callback_if_old and queue.qsize():
                    self._profiler.disable()
                    continue

                with lock:
                    precinct_assignment = self.assign_precincts_to_districts(
                        raw_districts, pixel_district_map)
            except Exception as e:
                self._profiler.disable()
                logging.exception(e)
                callback([], [], [])
                continue

            qsize = queue.qsize()
            if not callback_if_old and qsize:
                self._profiler.disable()
                continue

            callback(districts, fiducial_identity, fiducial_pos, post_callback,
                     (districts, raw_districts, precinct_assignment,
                      pixel_district_map),
                     bool(qsize))
            self._profiler.disable()

    def stop_thread(self):
        self._thread_queue.put('eof')
        self._thread.join()

    def request_reassignment(
            self, callback, ignore_if_scheduled=True,
            callback_if_old=False, current_fiducials=False):
        if ignore_if_scheduled and self._thread_queue.qsize():
            return

        fiducials = fiducial_ids = None
        if current_fiducials:
            fiducials = dict(self.fiducial_locations)
            fiducial_ids = dict(self.fiducial_ids)

        self._thread_queue.put(
            (callback, callback_if_old, fiducials, fiducial_ids))

    def set_precincts(self, precincts):
        """Adds the precincts to be used by the mapping.

        Must be called only (or every time) after :attr:`screen_size` is set.

        :param precincts: List of :class:`distopia.precinct.Precinct`
            instances.
        """
        w, h = self.screen_size
        self.precincts = list(precincts)
        colliders = self.precinct_colliders = [
            PolygonCollider(points=precinct.boundary, cache=True) for
            precinct in precincts]

        precinct_indices = self.precinct_indices = []
        for i, (precinct, collider) in enumerate(zip(precincts, colliders)):
            x1, y1, x2, y2 = collider.bounding_box()
            precinct_values = np.zeros(
                (x2 - x1 + 1, y2 - y1 + 1), dtype=np.uint8)
            precinct_indices.append((x1, y1, x2 + 1, y2 + 1, precinct_values))

            for x, y in collider.get_inside_points():
                if 0 <= x < w and 0 <= y < h:
                    precinct_values[x - x1, y - y1] = 1

    def add_fiducial(self, location, identity):
        """Adds a new fiducial at ``location``.

        :param location: The fiducial location ``(x, y)``.
        :return: The (assigned) ID of the fiducial.
        """
        i = self._fiducial_count
        self._fiducial_count += 1

        x, y = location
        w, h = self.screen_size
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)

        with self.thread_lock:
            self.fiducial_locations[i] = x, y  # queue safe
            self.fiducial_ids[i] = identity
        return i

    def move_fiducial(self, fiducial, location):
        """Moves ``fiducial`` from its previous location to a new location.

        :param fiducial: ``fiducial`` ID as returned by :meth:`add_fiducial`.
        :param location: The new fiducial location ``(x, y)``.
        """
        x, y = location
        w, h = self.screen_size
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)

        self.fiducial_locations[fiducial] = x, y  # queue safe

    def remove_fiducial(self, fiducial):
        """Removes ``fiducial`` from the diagram.

        :param fiducial: ``fiducial`` ID as returned by :meth:`add_fiducial`.
        """
        with self.thread_lock:
            del self.fiducial_locations[fiducial]
            del self.fiducial_ids[fiducial]

    def get_fiducials(self):
        """Returns a dict of the fiducials, keys are their ID.
        """
        return self.fiducial_locations

    def get_fiducial_ids(self):
        return self.fiducial_ids

    def get_fiducial_district(self, fiducial):
        """The district under the fiducial.

        :param fiducial: ``fiducial`` ID as returned by :meth:`add_fiducial`.
        :return: The district under the fiducial.
        """
        x, y = map(int, self.fiducial_locations[fiducial])
        i = self.pixel_district_map[x, y]
        return self._raw_districts[i]

    def get_pos_district(self, pos):
        """The district under the fiducial.

        :param pos: position.
        :return: The district under the position, or None if none.
        """
        x, y = map(int, pos)
        i = self.pixel_district_map[x, y]
        if i == 2 ** 8 - 1:
            return None
        return self._raw_districts[i]

    def assign_precincts_to_districts(self, districts, pixel_district_map):
        """Uses the pre-computed precinct and district maps and assigns
        all the precincts to districts.
        """
        precincts = self.precincts
        precinct_assignment = [[] for _ in districts]

        bins = np.empty((len(districts), ), dtype=np.uint64)
        for i, (precinct, (x0, y0, x1, y1, mask), collider) in enumerate(
                zip(precincts, self.precinct_indices, self.precinct_colliders)):
            bins[:] = 0
            district_i = collider.get_arg_max_count(
                pixel_district_map[x0:x1, y0:y1],
                mask, bins, len(districts),
                x1 - x0, y1 - y0, 2 ** 8 - 1)
            if district_i == 2 ** 8 - 1:
                continue

            precinct_assignment[district_i].append(precinct)
        return precinct_assignment

    def compute_district_pixels(self, fiducials, fiducials_identity):
        """Computes the assignment of pixels to districts and creates the
        associated districts.
        """
        vor = Voronoi(fiducials)
        regions, vertices = self.voronoi_finite_polygons_2d(vor)

        assert len(regions) <= 2 ** 8 - 2
        w, h = self.screen_size
        pixel_district_map = np.ones((w, h), dtype=np.uint8) * (2 ** 8 - 1)

        raw_districts = districts = []
        for i, region_indices in enumerate(regions):
            poly = list(map(float, vertices[region_indices].reshape((-1, ))))
            collider = PolygonCollider(points=poly, cache=True)
            collider.mark_pixels(pixel_district_map, w, h, i)

            district = District()
            district.name = str(i)
            district.boundary = poly
            district.collider = collider
            district.identity = fiducials_identity[i]

            districts.append(district)

        return districts, raw_districts, pixel_district_map

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

            # get all vertices that already exists
            temp_vertices = [
                [(None, None), None, v, new_vertices[v]] for
                v in vertices if v >= 0]

            # all the ridges that make the region
            for p2, v1, v2 in ridges:
                # if the second point of the ridge is outside the finite area,
                # one and only one of the vertex indices will be -1
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    assert v2 >= 0
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                # normal, facing to the left of the line between p1 - p2
                n = np.array([-t[1], t[0]])

                # find midpoint between the two points
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                # find whether the normal points in the same direction as the
                # line made by the center of the voronoi points with midpoint.
                # If it faces the opposite direction, reorient to face from
                # the center to midpoint direction (i.e. away from the center)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                # add a point very far from the last point
                far_point = vor.vertices[v2] + direction * radius
                temp_vertices.append(
                    [vor.vertices[v2], direction, None, far_point])

            # finish
            temp_vertices = self.fix_voronoi_infinite_regions(
                temp_vertices, w - 1, h - 1)
            new_region = self.add_missing_polygon_points(
                temp_vertices, w - 1, h - 1, new_vertices)
            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices)

    def fix_voronoi_infinite_regions(
            self, vertices, w_max, h_max):
        vs = np.asarray([v[3] for v in vertices])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        vertices = [vertices[i] for i in np.argsort(angles)]

        for i, ((x1, y1), direction, v, (x2, y2)) in enumerate(vertices):
            if direction is None:
                continue

            slope = (y2 - y1) / (x2 - x1)
            offset = y1 - slope * x1
            # test if it intersects with top or bottom of screen
            if math.isclose(slope, 0):
                x2 = min(max(x2, 0), w_max)
            elif not math.isfinite(slope):
                y2 = min(max(y2, 0), h_max)
            else:
                x_top = (h_max - offset) / slope  # set y to h_max, solve x
                x_bot = (0 - offset) / slope  # set y to 0, solve x
                y_left = slope * 0 + offset  # set x to 0, solve y
                y_right = slope * w_max + offset  # set x to w_max, solve y
                if direction[0] > 0:
                    if direction[1] > 0:
                        if 0 <= x_top <= w_max:
                            y2 = h_max
                            x2 = x_top
                        elif 0 <= y_right <= h_max:
                            y2 = y_right
                            x2 = w_max
                    else:
                        if 0 <= x_bot <= w_max:
                            y2 = 0
                            x2 = x_bot
                        elif 0 <= y_right <= h_max:
                            y2 = y_right
                            x2 = w_max
                else:
                    if direction[1] > 0:
                        if 0 <= x_top <= w_max:
                            y2 = h_max
                            x2 = x_top
                        elif 0 <= y_left <= h_max:
                            y2 = y_left
                            x2 = 0
                    else:
                        if 0 <= x_bot <= w_max:
                            y2 = 0
                            x2 = x_bot
                        elif 0 <= y_left <= h_max:
                            y2 = y_left
                            x2 = 0

            x2 = min(max(x2, 0), w_max)
            y2 = min(max(y2, 0), h_max)
            vertices[i][3] = x2, y2
        return vertices

    def add_missing_polygon_points(self, vertices, w_max, h_max, new_vertices):
        region_verts_i = []
        for i, item in enumerate(vertices):
            if item[1] is None:
                region_verts_i.append(item[2])
            else:
                region_verts_i.append(len(new_vertices))
                new_vertices.append(item[3])

            if i == len(vertices) - 1:
                next_item = vertices[0]
            else:
                next_item = vertices[i + 1]

            # if one of the points is within the space, then we never add an
            # intermediate point because only two infinite points may contain
            # a corner between them
            if item[2] is not None or next_item[2] is not None:
                continue

            x1, y1 = item[3]
            x2, y2 = next_item[3]
            new_points = []
            if y1 == h_max and x2 == 0:
                new_points.append((0, h_max))
            elif x1 == 0 and y2 == 0:
                new_points.append((0, 0))
            elif y1 == 0 and x2 == w_max:
                new_points.append((w_max, 0))
            elif x1 == w_max and y2 == h_max:
                new_points.append((w_max, h_max))
            elif y1 == h_max and y2 == 0:
                new_points.append((0, h_max))
                new_points.append((0, 0))
            elif x1 == 0 and x2 == w_max:
                new_points.append((0, 0))
                new_points.append((w_max, 0))
            elif y1 == 0 and y2 == h_max:
                new_points.append((w_max, 0))
                new_points.append((w_max, h_max))
            elif x1 == w_max and x2 == 0:
                new_points.append((w_max, h_max))
                new_points.append((0, h_max))

            for new_point in new_points:
                region_verts_i.append(len(new_vertices))
                new_vertices.append(new_point)

        return region_verts_i
