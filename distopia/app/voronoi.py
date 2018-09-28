"""
Voronoi Kivy App
===================

Runs the voronoi GUI app.
"""
from itertools import cycle
import logging
import os
import cProfile, pstats, io
import numpy as np

from kivy.uix.widget import Widget
from kivy.app import App
from kivy.graphics.vertex_instructions import Line, Point, Mesh
from kivy.graphics.tesselator import Tesselator, WINDING_ODD, TYPE_POLYGONS
from kivy.graphics import Color
from matplotlib import colors as mcolors

import distopia
from distopia.app.geo_data import GeoData
from distopia.precinct import Precinct
from distopia.mapping.voronoi import VoronoiMapping

__all__ = ('VoronoiWidget', 'VoronoiApp')


class VoronoiWidget(Widget):
    """The widget through which we interact with the precincts and districts.
    """

    voronoi_mapping = None

    fiducial_graphics = {}

    district_graphics = []

    precinct_graphics = {}

    max_districts = 0

    table_mode = False

    _profiler = None

    align_mat = None

    def __init__(self, voronoi_mapping=None, max_districts=0, table_mode=False,
                 align_mat=None, **kwargs):
        super(VoronoiWidget, self).__init__(**kwargs)
        self.voronoi_mapping = voronoi_mapping
        self.fiducial_graphics = {}
        self.max_districts = max_districts
        self.table_mode = table_mode
        self.align_mat = align_mat
        self.district_graphics = []

        precinct_graphics = self.precinct_graphics = {}
        with self.canvas:
            for precinct in voronoi_mapping.precincts:
                assert len(precinct.boundary) >= 6
                tess = Tesselator()
                tess.add_contour(precinct.boundary)
                tess.tesselate(WINDING_ODD, TYPE_POLYGONS)

                graphics = [
                    Color(rgba=(0, 0, 0, 1))]
                for vertices, indices in tess.meshes:
                    graphics.append(
                        Mesh(
                            vertices=vertices, indices=indices,
                            mode="triangle_fan"))

                graphics.append(
                    Color(rgba=(0, 1, 0, 1)))
                graphics.append(
                    Line(points=precinct.boundary, width=1))
                precinct_graphics[precinct] = graphics

    def on_touch_up(self, touch):
        if self.table_mode:
            pass
        else:
            if not self.touch_mode_handle_up(touch.pos):
                return False
            self.recompute_voronoi()
            return True

    def touch_mode_handle_up(self, pos):
        if self.align_mat is not None:
            pos = tuple(
                np.dot(self.align_mat, np.array([pos[0], pos[1], 1]))[:2])

        x, y = pos
        for key, (x2, y2) in self.voronoi_mapping.get_fiducials().items():
            if ((x - x2) ** 2 + (y - y2) ** 2) ** .5 < 5:
                self.voronoi_mapping.remove_fiducial(key)

                for item in self.fiducial_graphics.pop(key):
                    self.canvas.remove(item)
                return True

        if len(self.voronoi_mapping.get_fiducials()) == self.max_districts:
            return False

        key = self.voronoi_mapping.add_fiducial(pos)

        with self.canvas:
            color = Color(rgba=(1, 0, 0, 1))
            point = Point(points=pos, pointsize=4)
            self.fiducial_graphics[key] = color, point
        return True

    def clear_voronoi(self):
        for district in self.voronoi_mapping.districts:
            for precinct in district.precincts:
                self.precinct_graphics[precinct][0].rgba = (0, 0, 0, 1)

        for item in self.district_graphics:
            self.canvas.remove(item)
        self.district_graphics = []

    def recompute_voronoi(self):
        if len(self.voronoi_mapping.get_fiducials()) <= 3:
            self.clear_voronoi()
            return

        self._profiler.enable()
        try:
            self.voronoi_mapping.compute_district_pixels()
        except Exception as e:
            logging.exception(e)
            self.clear_voronoi()
            self._profiler.disable()
            return

        self.voronoi_mapping.assign_precincts_to_districts()
        self._profiler.disable()

        colors = [
            mcolors.BASE_COLORS[c] for c in ('r', 'c', 'g', 'y', 'm', 'b')]
        for color, district in zip(cycle(colors), self.voronoi_mapping.districts):
            for precinct in district.precincts:
                self.precinct_graphics[precinct][0].rgb = color

        with self.canvas:
            self.district_graphics.append(Color(1, 1, 0, 1))
            for district in self.voronoi_mapping.districts:
                self.district_graphics.append(
                    Line(points=district.boundary, width=3))


class VoronoiApp(App):
    """The Kivy application that creates the GUI.
    """

    voronoi_mapping = None

    use_county_dataset = True

    geo_data = None

    precincts = []

    screen_size = (1900, 800)

    max_districts = 8

    table_mode = False

    _profiler = None

    alignment_filename = 'alignment.txt'

    def create_voronoi(self):
        """Loads and initializes all the data and voronoi mapping.
        """
        self.geo_data = geo_data = GeoData()
        if self.use_county_dataset:
            geo_data.dataset_name = 'County_Boundaries_24K'
        else:
            geo_data.dataset_name = 'WI_Election_Data_with_2017_Wards'
        geo_data.screen_size = self.screen_size

        geo_data.load_data()
        geo_data.generate_polygons()
        geo_data.scale_to_screen()

        self.voronoi_mapping = vor = VoronoiMapping()
        self.precincts = precincts = []

        for record, polygons in zip(geo_data.records, geo_data.polygons):
            precinct = Precinct(
                name=str(record[0]), boundary=polygons[0].reshape(-1).tolist())
            precincts.append(precinct)

        vor.set_precincts(precincts)

    def build(self):
        """Builds the GUI.
        """
        mat = None
        if self.alignment_filename:
            fname = os.path.join(
                os.path.dirname(distopia.__file__), 'data',
                self.alignment_filename)
            mat = np.loadtxt(fname, delimiter=',', skiprows=3)

        self.create_voronoi()
        widget = VoronoiWidget(
            voronoi_mapping=self.voronoi_mapping,
            max_districts=self.max_districts,
            table_mode=self.table_mode, align_mat=mat)
        self._profiler = widget._profiler = cProfile.Profile()
        return widget


if __name__ == '__main__':
    app = VoronoiApp()
    app.run()

    s = io.StringIO()
    ps = pstats.Stats(app._profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
