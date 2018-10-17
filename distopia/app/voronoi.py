"""
Voronoi Kivy App
===================

Runs the voronoi GUI app.
"""

from kivy.support import install_twisted_reactor
install_twisted_reactor()

from itertools import cycle
import logging
import os
import cProfile, pstats, io
import numpy as np
import json
import csv
from functools import cmp_to_key

from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.app import App
from kivy.graphics.vertex_instructions import Line, Point, Mesh
from kivy.graphics.tesselator import Tesselator, WINDING_ODD, TYPE_POLYGONS
from kivy.graphics import Color
import matplotlib.pyplot as plt
from kivy.clock import Clock
from kivy.graphics.context_instructions import \
    PushMatrix, PopMatrix, Rotate, Translate, Scale, MatrixInstruction

import distopia
from distopia.app.geo_data import GeoData
from distopia.precinct import Precinct
from distopia.mapping.voronoi import VoronoiMapping
from distopia.app.ros import RosBridge
from distopia.precinct.metrics import PrecinctHistogram
from distopia.district.metrics import DistrictHistogramAggregateMetric

__all__ = ('VoronoiWidget', 'VoronoiApp')


class VoronoiWidget(Widget):
    """The widget through which we interact with the precincts and districts.
    """

    voronoi_mapping = None

    fiducial_graphics = {}

    district_graphics = []

    precinct_graphics = {}

    colors = []

    fiducials_color = {}

    table_mode = False

    _profiler = None

    align_mat = None

    screen_offset = 0, 0

    touches = {}

    ros_bridge = None

    district_blocks_fid = None

    focus_block_fid = 8

    focus_block_logical_id = 8

    _has_focus = False

    district_metrics_fn = None

    state_metrics_fn = None

    metric_selection_width = 200

    def __init__(self, voronoi_mapping=None, table_mode=False,
                 align_mat=None, screen_offset=(0, 0), ros_bridge=None,
                 district_blocks_fid=None, focus_block_fid=0,
                 focus_block_logical_id=0, district_metrics_fn=None,
                 state_metrics_fn=None,
                 metric_selection_width=200, **kwargs):
        super(VoronoiWidget, self).__init__(**kwargs)
        self.voronoi_mapping = voronoi_mapping
        self.ros_bridge = ros_bridge
        self.district_blocks_fid = district_blocks_fid
        self.focus_block_fid = focus_block_fid
        self.focus_block_logical_id = focus_block_logical_id
        self.metric_selection_width = metric_selection_width

        self.fiducial_graphics = {}
        self.fiducials_color = {}
        self.colors = cycle(plt.get_cmap('tab10').colors)

        self.table_mode = table_mode
        self.align_mat = align_mat
        self.district_graphics = []
        self.district_metrics_fn = district_metrics_fn
        self.state_metrics_fn = state_metrics_fn
        self.screen_offset = screen_offset
        self.touches = {}

        with self.canvas.before:
            PushMatrix()
            Translate(*screen_offset)
        with self.canvas.after:
            PopMatrix()
        self.show_precincts()

    def show_precincts(self):
        precinct_graphics = self.precinct_graphics = {}
        with self.canvas:
            for precinct in self.voronoi_mapping.precincts:
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

    def add_fiducial(self, location, identity):
        fiducial = self.voronoi_mapping.add_fiducial(location, identity)
        if identity not in self.fiducials_color:
            self.fiducials_color[identity] = next(self.colors)
        return fiducial

    def remove_fiducial(self, fiducial, location):
        self.voronoi_mapping.remove_fiducial(fiducial)

    def handle_focus_block(self, pos=None):
        pass

    def on_touch_down(self, touch):
        if not self.table_mode:
            return False

        focus_id = self.focus_block_logical_id
        blocks_fid = self.district_blocks_fid
        if 'markerid' not in touch.profile or (
                touch.fid not in blocks_fid and touch.fid != focus_id):
            return False

        pos = self.align_touch(touch.pos)

        # handle focus block
        if touch.fid == focus_id:
            if self._has_focus:
                return True
            self.handle_focus_block(pos)
            self._has_focus = touch
            return True

        logical_id = blocks_fid.index(touch.fid)
        key = self.add_fiducial(pos, logical_id)

        with self.canvas:
            color = Color(rgba=(1, 1, 1, 1))
            point = Point(points=pos, pointsize=7)

        info = {'fid': touch.fid, 'fiducial_key': key, 'last_pos': pos,
                'graphics': (color, point), 'logical_id': logical_id}
        self.touches[touch.uid] = info

        self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def on_touch_move(self, touch):
        if not self.table_mode or touch.uid not in self.touches:
            return False

        info = self.touches[touch.uid]
        pos = self.align_touch(touch.pos)
        if info['last_pos'] == pos:
            return True

        info['last_pos'] = pos
        info['graphics'][1].points = pos

        self.voronoi_mapping.move_fiducial(info['fiducial_key'], pos)
        self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def on_touch_up(self, touch):
        if self.table_mode:
            if touch.uid not in self.touches:
                return False

            info = self.touches[touch.uid]
            self.canvas.remove(info['graphics'][0])
            self.canvas.remove(info['graphics'][1])

            self.remove_fiducial(
                info['fiducial_key'], self.align_touch(touch.pos))
            self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        else:
            if not self.touch_mode_handle_up(touch):
                return False

            self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def align_touch(self, pos):
        if self.align_mat is not None:
            pos = tuple(
                np.dot(self.align_mat, np.array([pos[0], pos[1], 1]))[:2])

        x0, y0 = self.screen_offset
        pos = pos[0] - x0, pos[1] - y0
        return pos

    def touch_mode_handle_up(self, touch):
        x, y = pos = self.align_touch(touch.pos)

        for key, (x2, y2) in self.voronoi_mapping.get_fiducials().items():
            if ((x - x2) ** 2 + (y - y2) ** 2) ** .5 < 5:
                self.remove_fiducial(key, pos)

                for item in self.fiducial_graphics.pop(key):
                    self.canvas.remove(item)
                return True

        fid_count = len(self.voronoi_mapping.get_fiducials())
        if fid_count == len(self.district_blocks_fid):
            return False

        key = self.add_fiducial(pos, fid_count)

        with self.canvas:
            color = Color(rgba=(1, 1, 1, 1))
            point = Point(points=pos, pointsize=4)
            self.fiducial_graphics[key] = color, point
        return True

    def voronoi_callback(self, *largs):
        def _callback(dt):
            self.process_voronoi_output(*largs)
        Clock.schedule_once(_callback)

    def clear_voronoi(self):
        for district in self.voronoi_mapping.districts:
            for precinct in district.precincts:
                self.precinct_graphics[precinct][0].rgba = (0, 0, 0, 1)

        for item in self.district_graphics:
            self.canvas.remove(item)
        self.district_graphics = []

    def process_voronoi_output(
            self, districts, fiducial_identity, fiducial_pos,
            post_callback=None, largs=(),
            data_is_old=False):
        if data_is_old:
            return

        if post_callback is not None:
            post_callback(*largs)

        self.district_metrics_fn(districts)
        state_metrics = self.state_metrics_fn(districts)

        fid_ids = [self.district_blocks_fid[i] for i in fiducial_identity]
        if self.ros_bridge is not None:
            self.ros_bridge.update_voronoi(
                fiducial_pos, fid_ids, fiducial_identity, districts,
                state_metrics)
        if not districts:
            self.clear_voronoi()
            return

        districts = self.voronoi_mapping.districts
        colors = self.fiducials_color
        for district in districts:
            color = colors[district.identity]
            for precinct in district.precincts:
                self.precinct_graphics[precinct][0].rgb = color

        for item in self.district_graphics:
            self.canvas.remove(item)
        self.district_graphics = []

        with self.canvas:
            self.district_graphics.append(Color(1, 1, 0, 1))
            for district in districts:
                self.district_graphics.append(
                    Line(points=district.boundary + district.boundary[:2],
                         width=2))


class VoronoiApp(App):
    """The Kivy application that creates the GUI.
    """

    voronoi_mapping = None

    ros_bridge = None

    use_county_dataset = True

    geo_data = None

    precincts = []

    screen_size = (1900, 800)

    table_mode = False

    _profiler = None

    alignment_filename = 'alignment.txt'

    screen_offset = 0, 0

    show_precinct_id = False

    district_blocks_fid = [0, 1, 2, 3, 4, 5, 6, 7]

    focus_block_fid = 8

    focus_block_logical_id = 8

    use_ros = False

    metrics = ['demographics', ]

    metric_selection_width = 200

    def create_district_metrics(self, districts):
        for district in districts:
            if 'demographics' in self.metrics:
                district.metrics['demographics'] = \
                    DistrictHistogramAggregateMetric(
                        district=district, name='demographics')

    def load_precinct_metrics(self):
        assert self.use_county_dataset
        if 'demographics' not in self.metrics:
            return

        geo_data = self.geo_data
        names = set(r[3] for r in geo_data.records)
        names = {v: v for v in names}
        names['Saint Croix'] = 'St. Croix'

        fname = os.path.join(
            os.path.dirname(distopia.__file__), 'data',
            'County_Boundaries_24K', 'demographics.csv')
        with open(fname) as fh:
            reader = csv.reader(fh)
            header = next(reader)

            data = {}
            for row in reader:
                data[row[0]] = list(map(int, row[1:]))

        for precinct, record in zip(self.precincts, geo_data.records):
            name = names[record[3]]
            precinct.metrics['demographics'] = PrecinctHistogram(
                name='demographics', labels=header, data=data[name])

    def create_state_metrics(self, districts):
        return []

    def create_voronoi(self):
        """Loads and initializes all the data and voronoi mapping.
        """
        self.geo_data = geo_data = GeoData()
        if self.use_county_dataset:
            geo_data.dataset_name = 'County_Boundaries_24K'
        else:
            geo_data.dataset_name = 'WI_Election_Data_with_2017_Wards'
            geo_data.source_coordinates = ''

        geo_data.screen_size = self.screen_size
        try:
            geo_data.load_npz_data()
        except FileNotFoundError:
            geo_data.load_data()
            geo_data.generate_polygons()
            geo_data.scale_to_screen()
            geo_data.smooth_vertices()

        self.voronoi_mapping = vor = VoronoiMapping()
        vor.screen_size = self.screen_size
        self.precincts = precincts = []

        for i, (record, polygons) in enumerate(
                zip(geo_data.records, geo_data.polygons)):
            precinct = Precinct(
                name=str(record[0]), boundary=polygons[0].reshape(-1).tolist(),
                identity=i, location=polygons[0].mean(axis=0).tolist())
            precincts.append(precinct)

        vor.set_precincts(precincts)

    def show_precinct_labels(self, widget):
        for i, precinct in enumerate(self.precincts):
            label = Label(
                text=str(precinct.identity), center=precinct.location,
                font_size=20)
            widget.add_widget(label)

    def load_config(self):
        keys = ['use_county_dataset', 'screen_size',
                'table_mode', 'alignment_filename', 'screen_offset',
                'show_precinct_id', 'focus_block_fid',
                'focus_block_logical_id', 'district_blocks_fid', 'use_ros',
                'metrics']

        fname = os.path.join(
            os.path.dirname(distopia.__file__), 'data', 'config.json')
        if not os.path.exists(fname):
            config = {key: getattr(self, key) for key in keys}
            with open(fname, 'w') as fp:
                json.dump(config, fp, indent=2, sort_keys=True)

        with open(fname, 'r') as fp:
            for key, val in json.load(fp).items():
                setattr(self, key, val)

        config = {key: getattr(self, key) for key in keys}
        with open(fname, 'w') as fp:
            json.dump(config, fp, indent=2, sort_keys=True)

    def build(self):
        """Builds the GUI.
        """
        self.load_config()

        mat = None
        if self.alignment_filename:
            fname = os.path.join(
                os.path.dirname(distopia.__file__), 'data',
                self.alignment_filename)
            mat = np.loadtxt(fname, delimiter=',', skiprows=3)

        self.create_voronoi()
        self.load_precinct_metrics()
        if self.use_ros:
            self.ros_bridge = RosBridge()
        widget = VoronoiWidget(
            voronoi_mapping=self.voronoi_mapping,
            table_mode=self.table_mode, align_mat=mat,
            screen_offset=self.screen_offset, ros_bridge=self.ros_bridge,
            district_blocks_fid=self.district_blocks_fid,
            focus_block_fid=self.focus_block_fid,
            focus_block_logical_id=self.focus_block_logical_id,
            district_metrics_fn=self.create_district_metrics,
            state_metrics_fn=self.create_state_metrics,
            metric_selection_width=self.metric_selection_width)

        if self.show_precinct_id:
            self.show_precinct_labels(widget)
        self._profiler = widget._profiler = cProfile.Profile()
        return widget


if __name__ == '__main__':
    app = VoronoiApp()
    try:
        app.run()
    finally:
        app.voronoi_mapping.stop_thread()
        if app.ros_bridge:
            app.ros_bridge.stop_threads()

    # s = io.StringIO()
    # ps = pstats.Stats(app._profiler, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # print(s.getvalue())
