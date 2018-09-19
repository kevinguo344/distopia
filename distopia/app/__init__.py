"""Distopia Application
========================

Runs the application.
"""

import os.path
import distopia
import shapefile
from pyproj import Proj, transform
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from distopia.precinct import Precinct
from distopia.mapping.voronoi import VoronoiMapping
import math
from kivy.uix.widget import Widget
from kivy.app import App
from kivy.graphics.vertex_instructions import Line, Point, Mesh
from kivy.graphics import Color
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon


def read_wisconsin_data(
        dataset='WI_Election_Data_with_2017_Wards', screen_size=(1920, 1080)):
    # https://elections.wi.gov/elections-voting/statistics
    # https://github.com/nvkelso/election-geodata/issues/2
    # https://doi.org/10.7910/DVN/NH5S2I
    # https://hub.arcgis.com/datasets/f0532acbe3304e20a11dbd598c5654f7_0
    # https://elections.wi.gov/elections-voting/results/2016/fall-general
    # https://elections.wi.gov/sites/default/files/11.4.14%20Election%20Results-all%20offices-w%20x%20w%20report.pdf
    # https://www.cityofmadison.com/planning/unit_planning/map_aldermanic/ald_dist_8x10.pdf
    # wards are precincts?
    data_path = os.path.join(
        os.path.dirname(distopia.__file__), 'data', dataset, dataset)

    shp = shapefile.Reader(data_path)
    fields = [f[0] for f in shp.fields if not isinstance(f, tuple)]
    district_i = fields.index('CON')
    shapes = shp.shapes()
    records = shp.records()

    assert len(records)
    assert len(records[0]) == len(fields)

    crs_wgs = Proj(init='epsg:4326')
    crs_wisconsin = Proj(init='epsg:3857')

    polygons = []
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for record, shape in zip(records, shapes):
        if len(shape.parts) == 1:
            arr = np.array(
                [transform(crs_wgs, crs_wisconsin, lon, lat) for
                 lon, lat in shape.points])

            max_x = max(max_x, np.max(arr[:, 0]))
            max_y = max(max_y, np.max(arr[:, 1]))
            min_x = min(min_x, np.min(arr[:, 0]))
            min_y = min(min_y, np.min(arr[:, 1]))

            poly_list = [arr,]
        else:
            poly_list = []
            for i, s in enumerate(shape.parts):
                if i < len(shape.parts) - 1:
                    e = shape.parts[i + 1] - 1
                else:
                    e = len(shape.points)

                arr = np.array([
                    transform(crs_wgs, crs_wisconsin, lon, lat) for
                    lon, lat in shape.points[s:e + 1]])

                max_x = max(max_x, np.max(arr[:, 0]))
                max_y = max(max_y, np.max(arr[:, 1]))
                min_x = min(min_x, np.min(arr[:, 0]))
                min_y = min(min_y, np.min(arr[:, 1]))

                poly_list.append(arr)
        polygons.append(poly_list)

    if screen_size is not None:
        width, height = map(float, screen_size)
        # normalize to the screen size
        max_x, max_y = math.ceil(max_x), math.ceil(max_y)
        min_x, min_y = math.floor(min_x), math.floor(min_y)
        x_ratio = width / (max_x - min_x)
        y_ratio = height / (max_y - min_y)
        ratio = min(x_ratio, y_ratio)

        for precinct_polygons in polygons:
            for polygon in precinct_polygons:
                polygon[:, 0] -= min_x
                polygon[:, 0] *= ratio
                polygon[:, 1] -= min_y
                polygon[:, 1] *= ratio

    return polygons, records, fields


def make_mapping(precinct_data):
    vor = VoronoiMapping()
    precincts = []
    for name, polygon in precinct_data.items():
        precinct = Precinct(name=name, boundary=polygon.reshape(-1).tolist())
        precincts.append(precinct)
    vor.set_precincts(precincts)
    return vor


def plot_wards():
    colors = cycle(['r', 'g', 'b', 'y'])
    precinct_polygons, records, fields = read_wisconsin_data(screen_size=None)
    for record, polygons in zip(records, precinct_polygons):
        color = next(colors)
        polygon = polygons[0]
        plt.fill(polygons[0][:, 0], polygons[0][:, 1], color, edgecolor='k')
        if record[6] != 'Madison':
            continue
        plt.gca().annotate(
            record[9] + ' ' + record[12], (np.mean(polygon[:, 0]), np.mean(polygon[:, 1])),
            color='w', weight='bold', fontsize=6, ha='center', va='center')

        # for polygon in polygons[1:]:
        #     plt.fill(polygon[:, 0], polygon[:, 1], 'w')
            #plt.gca().add_patch(Polygon(polygon))
    plt.show()


class VoronoiWidget(Widget):

    vor = None

    fiducial_graphics = []

    district_graphics = []

    precinct_graphics = {}

    def __init__(self, vor=None, **kwargs):
        super(VoronoiWidget, self).__init__(**kwargs)
        self.fiducials = {}
        self.vor = vor

        precinct_graphics = self.precinct_graphics = {}
        with self.canvas:
            for precinct in vor.precincts:
                points_x = precinct.boundary[0::2]
                points_y = precinct.boundary[1::2]
                # points_x.append(points_x[0])
                # points_y.append(points_y[0])
                padding = [0, ] * len(points_x)
                vertices = [
                    val for item in zip(points_x, points_y, padding, padding)
                    for val in item]
                indices = list(range(len(points_x)))

                c = Color(rgba=(0, 0, 0, 1))
                m = Mesh(vertices=vertices, indices=indices)
                m.mode = 'triangle_fan'
                c2 = Color(rgba=(0, 1, 0, 1))
                line = Line(points=precinct.boundary, width=1)
                precinct_graphics[precinct] = (c, m, line, c2)

    def on_touch_up(self, touch):
        x, y = touch.pos
        for i, (x2, y2) in enumerate(self.vor.get_fiducials()):
            if ((x - x2) ** 2 + (y - y2) ** 2) ** .5 < 5:
                self.vor.remove_fiducial(i)

                for item in self.fiducial_graphics[i]:
                    self.canvas.remove(item)
                del self.fiducial_graphics[i]
                break
        else:
            i = self.vor.add_fiducial(touch.pos)

            with self.canvas:
                color = Color(rgba=(1, 0, 0, 1))
                point = Point(points=touch.pos, pointsize=4)
                self.fiducial_graphics.append((color, point))

        for item in self.district_graphics:
            self.canvas.remove(item)
        district_graphics = self.district_graphics = []

        if len(self.vor.get_fiducials()) <= 3:
            return True

        import time
        t0 = time.clock()
        print('initialo')
        self.vor.compute_district_pixels()
        print('init2', time.clock() - t0)
        self.vor.assign_precincts_to_districts()
        print('init3', time.clock() - t0)

        with self.canvas:
            district_graphics.append(Color(rgba=(0, 0, 1, 1)))
            for district in self.vor.districts:
                district_graphics.append(Line(points=district.boundary, width=3))

        for color, district in zip(cycle(mcolors.BASE_COLORS.values()), self.vor.districts):
            for precinct in district.precincts:
                self.precinct_graphics[precinct][0].rgb = color


class VoronoiApp(App):

    vor = None

    def build(self):
        precinct_polygons, records, fields = read_wisconsin_data()

        precinct_data = {}
        for record, polygons in zip(records, precinct_polygons):
            key = '{}_{}_{}'.format(record[1], *record[11:13])
            assert key not in precinct_data
            precinct_data[key] = polygons[0]

        self.vor = vor = make_mapping(precinct_data)

        widget = VoronoiWidget(vor=vor)
        return widget


if __name__ == '__main__':
    VoronoiApp().run()
