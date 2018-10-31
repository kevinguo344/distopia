'''
Collider
========

The collider module contains classes which can be used to test membership
of a point in some space. See individual class documentation for details.

.. image:: _static/Screenshot.png
    :align: right

To use it you first have to cythonize the code. To do that, cd to the directory
containing collider.pyx and::

    python setup.py build_ext --inplace
'''

__all__ = ('PolygonCollider', 'fill_voronoi_diagram')


cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
cdef extern from "math.h":
    double round(double val)
    double floor(double val)
    double ceil(double val)
    double pow(double x, double y)
    double sqrt(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_voronoi_diagram(
    np.ndarray[np.uint8_t, ndim=2] pixels, int w, int h,
    np.ndarray[np.float64_t, ndim=2] sites, np.ndarray[np.uint8_t] site_ids):
    cdef int x, y, i, i_min
    cdef double dist, dist_min
    if not len(site_ids):
        raise ValueError('No sites specified')

    for x in range(w):
        for y in range(h):
            dist_min = sqrt(pow(x - sites[0, 0], 2) + pow(y - sites[0, 1], 2))
            i_min = 0

            for i in range(1, len(site_ids)):
                dist = sqrt(pow(x - sites[i, 0], 2) + pow(y - sites[i, 1], 2))
                if dist < dist_min:
                    dist_min = dist
                    i_min = i

            pixels[x, y] = site_ids[i_min]


cdef class PolygonCollider(object):
    ''' PolygonCollider checks whether a point is within a polygon defined by a
    list of corner points.

    ``rect`` is (x, y, width, height)

    Based on http://alienryderflex.com/polygon/

    For example, a simple triangle::

        >>> collider = PolygonCollider([10., 10., 20., 30., 30., 10.],
        ... cache=True)
        >>> (0.0, 0.0) in collider
        False
        >>> (20.0, 20.0) in collider
        True

    The constructor takes a list of x,y points in the form of [x1,y1,x2,y2...]
    as the points argument. These points define the corners of the
    polygon. The boundary is linearly interpolated between each set of points.
    The x, and y values must be floating points.
    The cache argument, if True, will calculate membership for all the points
    so when collide_point is called it'll just be a table lookup.
    '''

    cdef double *cpoints
    cdef double *cconstant
    cdef double *cmultiple
    cdef char *cspace
    # bounding box of the polygon (inclusive, width/height add 1)
    # this is an integer
    cdef double min_x
    cdef double max_x
    cdef double min_y
    cdef double max_y
    # size of the cached array after x/y_start
    cdef int width
    cdef int height
    # offset in bounding box where the cached buffer starts
    cdef int x_start
    cdef int y_start
    # offset in coordinates relative to the bounding box
    cdef int x_offset
    cdef int y_offset
    cdef int empty
    cdef object rect
    # num polygon points
    cdef int count

    @cython.cdivision(True)
    def __cinit__(self, points, cache=False, rect=None, **kwargs):
        cdef int length = len(points)
        if length % 2:
            raise IndexError('Odd number of points provided')
        if length < 6:
            self.cpoints = NULL
            return

        cdef int count = length / 2
        self.count = count
        self.rect = rect
        self.cpoints = <double *>malloc(length * cython.sizeof(double))
        self.cconstant = <double *>malloc(count * cython.sizeof(double))
        self.cmultiple = <double *>malloc(count * cython.sizeof(double))
        cdef double *cpoints = self.cpoints
        cdef double *cconstant = self.cconstant
        cdef double *cmultiple = self.cmultiple
        self.cspace = NULL
        if cpoints is NULL or cconstant is NULL or cmultiple is NULL:
            raise MemoryError()

        self.min_x = floor(min(points[0::2]))
        self.max_x = ceil(max(points[0::2]))
        self.min_y = floor(min(points[1::2]))
        self.max_y = ceil(max(points[1::2]))

        cdef int i_x, i_y, j_x, j_y, i
        cdef int j = count - 1, odd, x, y
        for i in range(length):
            cpoints[i] = points[i]

        if cache:
            for i in range(count):
                cpoints[2 * i] -= self.min_x
                cpoints[2 * i + 1] -= self.min_y

        for i in range(count):
            i_x = i * 2
            i_y = i_x + 1
            j_x = j * 2
            j_y = j_x + 1
            if cpoints[j_y] == cpoints[i_y]:
                cconstant[i] = cpoints[i_x]
                cmultiple[i] = 0.
            else:
                cconstant[i] = (cpoints[i_x] - cpoints[i_y] * cpoints[j_x] /
                               (cpoints[j_y] - cpoints[i_y]) +
                               cpoints[i_y] * cpoints[i_x] /
                               (cpoints[j_y] - cpoints[i_y]))
                cmultiple[i] = ((cpoints[j_x] - cpoints[i_x]) /
                               (cpoints[j_y] - cpoints[i_y]))
            j = i

        cdef int x_, y_
        if cache:
            self.width = int(self.max_x - self.min_x + 1.)
            self.height = int(self.max_y - self.min_y + 1.)
            self.x_start = self.y_start = 0
            self.empty = 0

            if rect is not None:
                x_off, y_off, w_, h_ = rect
                w_ = int(ceil(w_))
                h_ = int(ceil(h_))
                self.x_offset = int(round(x_off - self.min_x))
                self.y_offset = int(round(y_off - self.min_y))

                # the rect is contained in polygon
                if self.x_offset >= 0:
                    self.width = max(min(w_, self.width - self.x_offset), 0)
                    self.x_start = self.x_offset
                else:
                    self.width = max(min(w_ + self.x_offset, self.width), 0)
                    self.x_start = 0

                if self.y_offset >= 0:
                    self.height = max(min(h_, self.height - self.y_offset), 0)
                    self.y_start = self.y_offset
                else:
                    self.height = max(min(h_ + self.y_offset, self.height), 0)
                    self.y_start = 0

            if not self.height or not self.width:
                self.empty = 1
                self.cspace = <char *>malloc(cython.sizeof(char))
                if self.cspace is NULL:
                    raise MemoryError()
                return

            self.cspace = <char *>malloc(self.height * self.width * cython.sizeof(char))
            if self.cspace is NULL:
                raise MemoryError()

            for y_ in range(self.y_start, self.y_start + self.height):
                y = y_ - self.y_start
                for x_ in range(self.x_start, self.x_start + self.width):
                    x = x_ - self.x_start
                    j = count - 1
                    odd = 0
                    for i in range(count):
                        i_y = i * 2 + 1
                        j_y = j * 2 + 1
                        if (cpoints[i_y] < y_ <= cpoints[j_y] or
                            cpoints[j_y] < y_ <= cpoints[i_y]):
                            odd ^= y_ * cmultiple[i] + cconstant[i] < x_
                        j = i
                    self.cspace[y * self.width + x] = odd

    def __dealloc__(self):
        free(self.cpoints)
        free(self.cconstant)
        free(self.cmultiple)
        free(self.cspace)

    @cython.cdivision(True)
    cpdef collide_point(self, double x, double y):
        cdef double *points = self.cpoints
        if points is NULL or not (self.min_x <= x <= self.max_x and
                                  self.min_y <= y <= self.max_y):
            return False

        cdef int x_, y_
        if self.cspace is not NULL:
            if self.empty:
                return False

            # x/y relative to bounding box
            x_ = int(round(x - self.min_x))
            y_ = int(round(y - self.min_y))
            if not (self.x_start <= x_ < self.x_start + self.width and self.y_start <= y_ < self.y_start + self.height):
                return False
            return self.cspace[(y_ - self.y_start) * self.width + x_ - self.x_start]

        cdef int j = self.count - 1, odd = 0, i, i_y, j_y, i_x, j_x
        for i in range(self.count):
            i_y = i * 2 + 1
            j_y = j * 2 + 1
            if points[i_y] < y <= points[j_y] or points[j_y] < y <= points[i_y]:
                odd ^= y * self.cmultiple[i] + self.cconstant[i] < x
            j = i
        return odd

    def __contains__(self, point):
        return self.collide_point(*point)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mark_pixels_u8(self, np.ndarray[np.uint8_t, ndim=2] table, int w, int h,
                       np.uint8_t value):
        cdef int x, y, x_offset, y_offset
        if self.cspace is NULL:
            raise TypeError('This method can only be called if cache was True')
        if self.empty:
            return
        if self.rect is None:
            raise ValueError('Can only be called if rect was specified')
        if w != self.rect[2] or h != self.rect[3]:
            raise ValueError('The w,h does not match the w/h provided in rect')

        if self.x_offset >= 0:
            x_offset = 0
        else:
            x_offset = -self.x_offset

        if self.y_offset >= 0:
            y_offset = 0
        else:
            y_offset = -self.y_offset

        for x in range(self.width):
            for y in range(self.height):
                if self.cspace[y * self.width + x]:
                    table[x + x_offset, y + y_offset] = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mark_pixels_u16(
            self, np.ndarray[np.uint16_t, ndim=2] table, int w, int h, np.uint16_t value):
        cdef int x, y, x_offset, y_offset
        if self.cspace is NULL:
            raise TypeError('This method can only be called if cache was True')
        if self.empty:
            return
        if self.rect is None:
            raise ValueError('Can only be called if rect was specified')
        if w != self.rect[2] or h != self.rect[3]:
            raise ValueError('The w,h does not match the w/h provided in rect')

        if self.x_offset >= 0:
            x_offset = 0
        else:
            x_offset = -self.x_offset

        if self.y_offset >= 0:
            y_offset = 0
        else:
            y_offset = -self.y_offset

        for x in range(self.width):
            for y in range(self.height):
                if self.cspace[y * self.width + x]:
                    table[x + x_offset, y + y_offset] = value

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_arg_max_count(
            np.ndarray[np.uint8_t, ndim=2] table,
            np.ndarray[np.uint8_t, ndim=2] mask,
            np.ndarray[np.uint64_t, ndim=1] bins, int num_bins, int w, int h,
            unsigned char none_val):
        cdef int x, y, i, best_i
        cdef unsigned char val
        cdef np.uint64_t count_val, max_val

        for x in range(w):
            for y in range(h):
                if mask[x, y]:
                    val = table[x, y]
                    if val != none_val:
                        bins[val] += 1

        max_val = 0
        best_i = -1
        for i in range(num_bins):
            count_val = bins[i]
            if count_val > max_val:
                best_i = i
                max_val = count_val

        if best_i == -1:
            return none_val
        return best_i

    def get_inside_points(self):
        '''Returns a list of all the points that are within the polygon.
        '''
        cdef int x, y
        cdef list points = []

        if self.cspace is not NULL:
            if self.empty:
                return points

            for x in range(self.width):
                for y in range(self.height):
                    if self.cspace[y * self.width + x]:
                        points.append((
                            int(x + self.x_start + self.min_x),
                            int(y + self.y_start + self.min_y),
                        ))
            return points

        for x in range(int(floor(self.min_x)), int(ceil(self.max_x)) + 1):
            for y in range(int(floor(self.min_y)), int(ceil(self.max_y)) + 1):
                if self.collide_point(x, y):
                    points.append((x, y))
        return points

    def bounding_box(self):
        '''Returns the bounding box containing the polygon as 4 points
        (x1, y1, x2, y2), where x1, y1 is the lower left and x2, y2 is the
        upper right point of the rectangle.
        '''
        return int(self.min_x), int(self.min_y), int(self.max_x), int(self.max_y)

    def get_area(self):
        cdef int x, y
        cdef double count = 0

        if self.cspace is not NULL:
            if self.empty:
                return 0

            for x in range(self.width * self.height):
                if self.cspace[x]:
                    count += 1
            return count

        for x in range(int(floor(self.min_x)), int(ceil(self.max_x)) + 1):
            for y in range(int(floor(self.min_y)), int(ceil(self.max_y)) + 1):
                if self.collide_point(x, y):
                    count += 1
        return count

    def get_centroid(self):
        cdef double x = 0
        cdef double y = 0

        if self.cpoints is NULL:
            return 0, 0

        for i in range(self.count):
            x += self.cpoints[2 * i]
            y += self.cpoints[2 * i + 1]

        x = x / float(self.count)
        y = y / float(self.count)

        if self.cspace is not NULL:
            return x + self.min_x, y + self.min_y
        return x , y
