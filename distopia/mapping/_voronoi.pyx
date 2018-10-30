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
    cdef double min_x
    cdef double max_x
    cdef double min_y
    cdef double max_y
    cdef int width
    cdef int height
    cdef int count

    @cython.cdivision(True)
    def __cinit__(self, points, cache=False, **kwargs):
        cdef int length = len(points)
        if length % 2:
            raise IndexError('Odd number of points provided')
        if length < 6:
            self.cpoints = NULL
            return

        cdef int count = length / 2
        self.count = count
        self.cpoints = <double *>malloc(length * cython.sizeof(double))
        self.cconstant = <double *>malloc(count * cython.sizeof(double))
        self.cmultiple = <double *>malloc(count * cython.sizeof(double))
        cdef double *cpoints = self.cpoints
        cdef double *cconstant = self.cconstant
        cdef double *cmultiple = self.cmultiple
        self.cspace = NULL
        if cpoints is NULL or cconstant is NULL or cmultiple is NULL:
            raise MemoryError()

        self.min_x = min(points[0::2])
        self.max_x = max(points[0::2])
        self.min_y = min(points[1::2])
        self.max_y = max(points[1::2])
        cdef double min_x = floor(self.min_x), min_y = floor(self.min_y)
        cdef int i_x, i_y, j_x, j_y, i
        cdef int j = count - 1, odd, width, height, x, y
        for i in range(length):
            cpoints[i] = points[i]
        if cache:
            for i in range(count):
                cpoints[2 * i] -= min_x
                cpoints[2 * i + 1] -= min_y

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
        if cache:
            width = int(ceil(self.max_x) - min_x + 1.)
            self.width = width
            height = int(ceil(self.max_y) - min_y + 1.)
            self.height = height

            self.cspace = <char *>malloc(height * width * cython.sizeof(char))
            if self.cspace is NULL:
                raise MemoryError()
            for y in range(height):
                for x in range(width):
                    j = count - 1
                    odd = 0
                    for i in range(count):
                        i_y = i * 2 + 1
                        j_y = j * 2 + 1
                        if (cpoints[i_y] < y and cpoints[j_y] >= y or
                            cpoints[j_y] < y and cpoints[i_y] >= y):
                            odd ^= y * cmultiple[i] + cconstant[i] < x
                        j = i
                    self.cspace[y * width + x] = odd

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
        if self.cspace is not NULL:
            y -= floor(self.min_y)
            x -= floor(self.min_x)
            return self.cspace[int(y) * self.width + int(x)]

        cdef int j = self.count - 1, odd = 0, i, i_y, j_y, i_x, j_x
        for i in range(self.count):
            i_y = i * 2 + 1
            j_y = j * 2 + 1
            if (points[i_y] < y and points[j_y] >= y or
                points[j_y] < y and points[i_y] >= y):
                odd ^= y * self.cmultiple[i] + self.cconstant[i] < x
            j = i
        return odd

    def __contains__(self, point):
        return self.collide_point(*point)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _mark_pixels(
            self, int w, int h, int* _x0, int* _y0, int* _x1, int* _y1,
            int* _x_offset, int* _y_offset):
        cdef int x0, y0, x1, y1, x_offset, y_offset
        # this assumes that the table is the x, y coordinates which is between
        # 0 to w or h
        x_offset = int(floor(self.min_x))
        y_offset = int(floor(self.min_y))
        if x_offset >= 0:
            x0 = 0
            if x_offset >= w:
                x1 = x0
            else:
                x1 = min(self.width, w - x_offset)
        else:
            x0 = -x_offset
            if x0 >= self.width:
                x1 = x0
            else:
                x1 = min(w, self.width + x_offset) + x0

        if y_offset >= 0:
            y0 = 0
            if y_offset >= h:
                y1 = y0
            else:
                y1 = min(self.height, h - y_offset)
        else:
            y0 = -y_offset
            if y0 >= self.height:
                y1 = y0
            else:
                y1 = min(h, self.height + y_offset) + y0

        _x0[0] = x0
        _y0[0] = y0
        _x1[0] = x1
        _y1[0] = y1
        _x_offset[0] = x_offset
        _y_offset[0] = y_offset
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mark_pixels_u8(self, np.ndarray[np.uint8_t, ndim=2] table, int w, int h,
                       np.uint8_t value):
        cdef int x, y, x0, y0, x1, y1, x_offset, y_offset
        if self.cspace is NULL:
            raise TypeError('This method can only be called if cache was True')

        self._mark_pixels(w, h, &x0, &y0, &x1, &y1, &x_offset, &y_offset)

        for x in range(x0, x1):
            for y in range(y0, y1):
                if self.cspace[y * self.width + x]:
                    table[x + x_offset, y + y_offset] = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mark_pixels_u16(
            self, np.ndarray[np.uint16_t, ndim=2] table, int w, int h, np.uint16_t value):
        cdef int x, y, x0, y0, x1, y1, x_offset, y_offset
        if self.cspace is NULL:
            raise TypeError('This method can only be called if cache was True')

        self._mark_pixels(w, h, &x0, &y0, &x1, &y1, &x_offset, &y_offset)

        for x in range(x0, x1):
            for y in range(y0, y1):
                if self.cspace[y * self.width + x]:
                    table[x + x_offset, y + y_offset] = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_arg_max_count(
            self, np.ndarray[np.uint8_t, ndim=2] table,
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
        return (int(floor(self.min_x)), int(floor(self.min_y)),
                int(ceil(self.max_x)), int(ceil(self.max_y)))

    def get_area(self):
        cdef int x, y
        cdef double count = 0

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
