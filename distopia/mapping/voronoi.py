"""
Voronoi Mapping
===============
"""
from scipy.spatial import Voronoi
from distopia.district import District
from distopia.precinct import Precinct
from distopia.mapping._voronoi import PolygonCollider, fill_voronoi_diagram
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

    pixel_precinct_map = None

    precinct_indices = []

    _fiducial_count = 0

    _thread = None

    _thread_queue = None

    _profiler = None

    thread_lock = None

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

    def start_processing_thread(self):
        self._thread = thread = Thread(
            target=self.voronoi_thread_function)
        thread.start()

    def apply_voronoi(self):
        """Not thread safe."""
        fiducials = dict(self.fiducial_locations)
        fiducial_ids = dict(self.fiducial_ids)

        if len(fiducials) <= 3:
            return []

        fiducial_keys = list(fiducials.keys())
        fiducial_pos = [fiducials[key] for key in fiducial_keys]
        fiducial_identity = [fiducial_ids[key] for key in fiducial_keys]
        unique_ids = list(sorted(set(fiducial_identity)))

        # w, h = self.screen_size
        # pixel_district_map = np.empty((w, h), dtype=np.uint8)
        # fill_voronoi_diagram(
        #     pixel_district_map, w, h, np.array(fiducial_pos, dtype=np.float64),
        #     np.array(fiducial_identity, dtype=np.uint8))
        pixel_district_map = self.compute_district_pixels(
            np.asarray(fiducial_pos), fiducial_identity, unique_ids)
        precinct_assignment = self.assign_precincts_to_districts(
            len(unique_ids), pixel_district_map)
        districts, error = self.create_districts_from_assignment(
            precinct_assignment, unique_ids)
        if error:
            return []

        self.set_districts_boundary(districts)

        self.districts = districts
        self.pixel_district_map = pixel_district_map
        for district, precincts in zip(districts, precinct_assignment):
            district.assign_precincts(precincts)

        return districts

    def post_thread_computation_callback(
            self, districts, precinct_assignment, pixel_district_map):
        self.districts = districts
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
                    fiducial_ids = dict(self.fiducial_ids)

            if len(fiducials) <= 3:
                callback([], [], [])
                continue

            fiducial_keys = list(fiducials.keys())
            fiducial_pos = [fiducials[key] for key in fiducial_keys]
            fiducial_identity = [fiducial_ids[key] for key in fiducial_keys]
            unique_ids = list(sorted(set(fiducial_identity)))

            self._profiler.enable()
            try:
                pixel_district_map = self.compute_district_pixels(
                    np.asarray(fiducial_pos), fiducial_identity, unique_ids)
                if not callback_if_old and queue.qsize():
                    continue

                precinct_assignment = self.assign_precincts_to_districts(
                    len(unique_ids), pixel_district_map)
                if not callback_if_old and queue.qsize():
                    continue

                districts, error = self.create_districts_from_assignment(
                    precinct_assignment, unique_ids)
                if error:
                    if callback_if_old or not queue.qsize():
                        callback(districts, [], [], error)
                    continue

                self.set_districts_boundary(districts)

            except Exception as e:
                logging.exception(e)
                callback([], [], [])
                continue
            finally:
                self._profiler.disable()

            qsize = queue.qsize()
            if not callback_if_old and qsize:
                self._profiler.disable()
                continue

            callback(
                districts, fiducial_identity, fiducial_pos, [], post_callback,
                (districts, precinct_assignment, pixel_district_map),
                bool(qsize))
            self._profiler.disable()

    def stop_thread(self):
        if self._thread is not None:
            self._thread_queue.put('eof')
            self._thread.join()
        self._thread = None

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
        precincts = self.precincts = list(precincts)
        if len(precincts) < 2 ** 8 - 1:
            dtype = np.uint8
            f = 'mark_pixels_u8'
        elif len(precincts) < 2 ** 16 - 1:
            dtype = np.uint16
            f = 'mark_pixels_u16'
        else:
            raise ValueError('Too many precincts')

        self.pixel_precinct_map = pixel_precinct_map = np.ones(
            (w, h), dtype=dtype) * np.iinfo(dtype).max

        colliders = self.precinct_colliders = [
            PolygonCollider(
                points=precinct.boundary, cache=True, rect=(0, 0, w, h)) for
            precinct in precincts]

        precinct_indices = self.precinct_indices = []
        for i, (precinct, collider) in enumerate(zip(precincts, colliders)):
            getattr(collider, f)(pixel_precinct_map, w, h, i)

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

    def get_pos_district(self, pos):
        """The district under the fiducial.

        :param pos: position.
        :return: The district under the position, or None if none.
        """
        x, y = map(int, pos)
        d_i = self.pixel_district_map[x, y]
        p_i = self.pixel_precinct_map[x, y]

        if p_i == np.iinfo(self.pixel_precinct_map.dtype).max:
            return None

        return self.districts[d_i]

    @staticmethod
    def create_districts_from_assignment(precinct_assignment, unique_ids):
        districts = []
        for i in range(len(precinct_assignment)):
            district = District()
            district.name = str(unique_ids[i])
            district.identity = unique_ids[i]
            districts.append(district)

        district_map = {}
        for i, precincts in enumerate(precinct_assignment):
            for precinct in precincts:
                district_map[precinct] = i

        disconnected = []
        for precincts in precinct_assignment:
            if len(precincts) <= 1:
                continue

            disconnected = Precinct.find_disconnected_precincts(
                precincts, district_map)
            if disconnected:
                break

        return districts, disconnected

    def set_districts_boundary(self, districts):
        pass

    def assign_precincts_to_districts(self, n_districts, pixel_district_map):
        """Uses the pre-computed precinct and district maps and assigns
        all the precincts to districts.

        Returns list of precincts, per district. Indices correspond with the
        district identity index in `unique_ids` as filled into
        pixel_district_map.
        """
        precincts = self.precincts
        precinct_assignment = [[] for _ in range(n_districts)]

        bins = np.empty((n_districts, ), dtype=np.uint64)
        for i, (precinct, (x0, y0, x1, y1, mask), collider) in enumerate(
                zip(precincts, self.precinct_indices, self.precinct_colliders)):
            bins[:] = 0
            district_i = collider.get_arg_max_count(
                pixel_district_map[x0:x1, y0:y1],
                mask, bins, n_districts,
                x1 - x0, y1 - y0, 2 ** 8 - 1)

            if district_i == 2 ** 8 - 1:
                print('Got precinct under no district', precinct.identity)
                continue

            precinct_assignment[district_i].append(precinct)
        return precinct_assignment

    def compute_district_pixels(
            self, fiducials, fiducials_identity, unique_ids):
        """Computes the assignment of pixels to districts and creates the
        associated districts.

        fiducials and fiducials_identity must be sorted such that they
        correspond to each other. All ids must be in unique ids.
        pixel_district_map is filled in with the index of the district
        identity in unique_ids to make it 0-n-1.
        """
        t0 = time.clock()
        vor = Voronoi(fiducials)
        t1 = time.clock()
        regions, vertices = self.voronoi_finite_polygons_2d(vor)
        t2 = time.clock()
        assert len(regions) <= 2 ** 8 - 2
        w, h = self.screen_size
        pixel_district_map = np.ones((w, h), dtype=np.uint8) * (2 ** 8 - 1)

        colliders = []
        for i, region_indices in enumerate(regions):
            poly = list(map(float, vertices[region_indices].reshape((-1, ))))
            collider = PolygonCollider(points=poly, cache=True, rect=(0, 0, w, h))
            colliders.append(collider)

        t3 = time.clock()
        for i, (region_indices, collider) in enumerate(zip(regions, colliders)):
            idx = unique_ids.index(fiducials_identity[i])
            collider.mark_pixels_u8(pixel_district_map, w, h, idx)

        t4 = time.clock()
        # print("{:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}".format(t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return pixel_district_map

    def voronoi_finite_polygons_2d(self, vor):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        based on https://stackoverflow.com/a/20678647.

        Parameters
        ----------
        vor : Voronoi
            Input diagram

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
        orig_vertices = vor.vertices.tolist()
        new_vertices = list(orig_vertices)
        # list of tuple of indices in orig_vertices, each describing a ridge
        # made by a pair of returned vertices
        ridge_vertices = vor.ridge_vertices
        # list of tuples, each two indices in points. They describe a ridge
        # perpendicular to their line
        ridge_points = vor.ridge_points
        assert len(ridge_vertices) == len(ridge_points)
        # list of tuple, each a point passed by the user
        points = vor.points
        # for each point in points, it's the index in regions that contains it
        point_region = vor.point_region
        assert len(points) == len(point_region)
        # list of lists, each is indices in orig_vertices making the polygon
        regions = vor.regions
        assert len(points) == len([r for r in regions if r])

        center = vor.points.mean(axis=0)
        w, h = self.screen_size
        radius = 100 * max(w, h)

        # Construct a map containing all ridges for a given point
        # for every point it lists all counter points and vertices between them
        all_ridges = defaultdict(list)
        for (p1, p2), (v1, v2) in zip(ridge_points, ridge_vertices):
            # points p1, p2 will have the same vertices between them
            all_ridges[p1].append((p1, p2, v1, v2))
            all_ridges[p2].append((p1, p2, v1, v2))

        # compute the polygons for each region
        for p1, region in enumerate(point_region):
            # The region polygon. First get all vertices for the region within
            # the screen
            region_vertices = []
            for v in regions[region]:
                # skip infinite vertices
                if v < 0:
                    continue

                # skip vertices outside the screen
                # x, y = orig_vertices[v]
                # if not (0 <= x <= w - 1 and 0 <= y <= h - 1):
                #     continue
                region_vertices.append(
                    [(None, None), None, v, orig_vertices[v]])

            # all the ridges that make the region
            for p1_, p2_, v1, v2 in all_ridges[p1]:
                # if the second point of the ridge is infinite,
                # one and only one of the vertex indices will be -1
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    assert v2 >= 0
                    # x1, y1 = orig_vertices[v1]
                    # x2, y2 = orig_vertices[v2]
                    #
                    # # assume that if a vertex is outside the screen,
                    # # the other vertex must be on the other side of the p1-p2
                    # # perpendicular
                    # if not (0 <= x1 <= w - 1 and 0 <= y1 <= h - 1):
                    #     region_vertices.append(
                    #         [[x2, y2], [x1 - x2, y1 - y2], None, [x1, y1]])
                    # if not (0 <= x2 <= w - 1 and 0 <= y2 <= h - 1):
                    #     region_vertices.append(
                    #         [[x1, y1], [x2 - x1, y2 - y1], None, [x2, y2]])

                    continue
                # assume that if there's an infinite point, the pair will be
                # within the screen boundary
                assert v2 >= 0

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2_] - vor.points[p1_]  # tangent
                t /= np.linalg.norm(t)
                # normal, facing to the left of the line between p1 - p2
                n = np.array([-t[1], t[0]])

                # find midpoint between the two points
                midpoint = vor.points[[p1_, p2_]].mean(axis=0)
                # find whether the normal points in the same direction as the
                # line made by the center of the voronoi points with midpoint.
                # If it faces the opposite direction, reorient to face from
                # the center to midpoint direction (i.e. away from the center)
                # It will be the same for p1, and p2 when we encounter each
                # this ensures that we get the same far point for each
                direction = np.sign(np.dot(midpoint - center, n)) * n
                # add a point very far from the last point
                far_point = orig_vertices[v2] + direction * radius
                region_vertices.append(
                    [orig_vertices[v2], direction, None, far_point])

            # for (x1, y1), direction, v, (x2, y2) in temp_vertices:
            #     if v is not None:
            #         assert x2 >= 0 and y2 >= 0
            # finish

            vs = np.asarray([v[3] for v in region_vertices])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            region_vertices = [region_vertices[i] for i in np.argsort(angles)]

            region_verts_i = []
            for i, item in enumerate(region_vertices):
                if item[1] is None:
                    region_verts_i.append(item[2])
                else:
                    region_verts_i.append(len(new_vertices))
                    new_vertices.append(item[3])

            new_regions.append(region_verts_i)

        # region is sorted according to points order
        new_vertices = np.asarray(new_vertices)
        return new_regions, new_vertices
