"""Distopia utilities
=======================
"""
import math


class PolygonCollider(object):
    ''' Collide2DPoly checks whether a point is within a polygon defined by a
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

    points = []
    cconstant = []
    cmultiple = []
    cspace = None

    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    width = 0
    count = 0

    def __init__(self, points, cache=False, **kwargs):
        super(PolygonCollider, self).__init__(**kwargs)
        length = len(points)
        if length % 2:
            raise IndexError('Odd number of points provided')
        if length < 6:
            self.points = []
            return

        self.count = count = length // 2
        points = self.points = list(map(float, points))
        cconstant = self.cconstant = [0, ] * count
        cmultiple = self.cmultiple = [0, ] * count

        self.min_x = min(points[0::2])
        self.max_x = max(points[0::2])
        self.min_y = min(points[1::2])
        self.max_y = max(points[1::2])

        min_x = math.floor(self.min_x)
        min_y = math.floor(self.min_y)

        j = count - 1

        if cache:
            for i in range(count):
                points[2 * i] -= min_x
                points[2 * i + 1] -= min_y

        for i in range(count):
            i_x = i * 2
            i_y = i_x + 1
            j_x = j * 2
            j_y = j_x + 1
            if points[j_y] == points[i_y]:
                cconstant[i] = points[i_x]
                cmultiple[i] = 0.
            else:
                cconstant[i] = (
                    points[i_x] - points[i_y] * points[j_x] /
                    (points[j_y] - points[i_y]) +
                    points[i_y] * points[i_x] /
                    (points[j_y] - points[i_y]))

                cmultiple[i] = (
                    (points[j_x] - points[i_x]) / (points[j_y] - points[i_y]))
            j = i

        if cache:
            width = int(math.ceil(self.max_x) - min_x + 1.)
            self.width = width

            height = int(math.ceil(self.max_y) - min_y + 1.)
            self.cspace = [0] * (height * width)

            for y in range(height):
                for x in range(width):
                    j = count - 1
                    odd = 0
                    for i in range(count):
                        i_y = i * 2 + 1
                        j_y = j * 2 + 1
                        if (points[i_y] < y <= points[j_y] or
                                points[j_y] < y <= points[i_y]):
                            odd ^= y * cmultiple[i] + cconstant[i] < x
                        j = i
                    self.cspace[y * width + x] = odd

    def collide_point(self, x, y):
        points = self.points
        if not points or not (
                self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y):
            return False

        if self.cspace is not None:
            y -= math.floor(self.min_y)
            x -= math.floor(self.min_x)
            return self.cspace[int(y) * self.width + int(x)]

        j = self.count - 1
        odd = 0
        for i in range(self.count):
            i_y = i * 2 + 1
            j_y = j * 2 + 1
            if (points[i_y] < y <= points[j_y] or
                    points[j_y] < y <= points[i_y]):
                odd ^= y * self.cmultiple[i] + self.cconstant[i] < x
            j = i
        return odd

    def __contains__(self, point):
        return self.collide_point(*point)

    def get_inside_points(self):
        '''Returns a list of all the points that are within the polygon.
        '''
        points = []

        for x in range(
                int(math.floor(self.min_x)), int(math.ceil(self.max_x)) + 1):
            for y in range(
                    int(math.floor(self.min_y)),
                    int(math.ceil(self.max_y)) + 1):
                if self.collide_point(x, y):
                    points.append((x, y))
        return points

    def bounding_box(self):
        '''Returns the bounding box containing the polygon as 4 points
        (x1, y1, x2, y2), where x1, y1 is the lower left and x2, y2 is the
        upper right point of the rectangle.
        '''
        return (int(math.floor(self.min_x)), int(math.floor(self.min_y)),
                int(math.ceil(self.max_x)), int(math.ceil(self.max_y)))

    def get_area(self):
        count = 0.

        for x in range(
                int(math.floor(self.min_x)), int(math.ceil(self.max_x)) + 1):
            for y in range(
                    int(math.floor(self.min_y)),
                    int(math.ceil(self.max_y)) + 1):
                if self.collide_point(x, y):
                    count += 1
        return count

    def get_centroid(self):
        x = y = 0.

        if not self.points:
            return 0, 0

        for i in range(self.count):
            x += self.points[2 * i]
            y += self.points[2 * i + 1]

        x = x / float(self.count)
        y = y / float(self.count)

        if self.cspace:
            return x + self.min_x, y + self.min_y
        return x, y
