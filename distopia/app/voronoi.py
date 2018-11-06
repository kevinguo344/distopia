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
import math
from functools import cmp_to_key

from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.lang import Builder
from kivy.app import App
from kivy.graphics.vertex_instructions import Line, Point, Mesh
from kivy.graphics.tesselator import Tesselator, WINDING_ODD, TYPE_POLYGONS
from kivy.graphics import Color
import matplotlib.pyplot as plt
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.properties import NumericProperty
from kivy.graphics.context_instructions import \
    PushMatrix, PopMatrix, Rotate, Translate, Scale, MatrixInstruction
from kivy.uix.spinner import Spinner

import distopia
from distopia.app.voronoi_data import GeoData, MetricData
from distopia.precinct import Precinct
from distopia.mapping.voronoi import VoronoiMapping
from distopia.app.ros import RosBridge

__all__ = ('VoronoiWidget', 'VoronoiApp')


class GuiTouchClassSpinner(Spinner):

    district_blocks_fid = []

    focus_block_logical_id = 0

    fid_id = NumericProperty(0)

    def __init__(self, district_blocks_fid=[], focus_block_logical_id=0,
                 **kwargs):
        super(GuiTouchClassSpinner, self).__init__(**kwargs)
        self.district_blocks_fid = district_blocks_fid
        self.focus_block_logical_id = focus_block_logical_id

        values = []
        for district in district_blocks_fid:
            values.append('District {}'.format(district))
        values.append('Focus')

        self.values = values
        self.text = values[0]
        self.fbind('text', self.process_selection)
        self.process_selection()

    def process_selection(self, *largs):
        text = self.text
        if text == 'Focus':
            self.fid_id = self.focus_block_logical_id
        else:
            self.fid_id = int(text.split(' ')[-1])


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

    district_blocks_fid = []

    focus_block_fid = 8

    focus_block_logical_id = 8

    _has_focus = False

    district_metrics_fn = None

    state_metrics_fn = None

    show_voronoi_boundaries = False

    current_fid_id = None

    focus_gui_pos = None

    focus_metrics = []

    focus_metric_width = 100

    focus_metric_height = 100

    screen_size = (1920, 1080)

    focus_region_width = 0

    n_focus_rows = 0

    n_focus_cols = 0

    gui_touch_focus_buttons = None

    max_fiducials_per_district = 5

    def __init__(
        self, voronoi_mapping=None, table_mode=False, align_mat=None,
        screen_offset=(0, 0), ros_bridge=None, district_blocks_fid=None,
        focus_block_fid=0, focus_block_logical_id=0, district_metrics_fn=None,
        state_metrics_fn=None,
        show_voronoi_boundaries=False, focus_metrics=[],
        focus_metric_width=100, focus_metric_height=100,
            screen_size=(1920, 1080), max_fiducials_per_district=5, **kwargs):
        super(VoronoiWidget, self).__init__(**kwargs)
        self.voronoi_mapping = voronoi_mapping
        self.ros_bridge = ros_bridge
        self.district_blocks_fid = district_blocks_fid
        self.focus_block_fid = focus_block_fid
        self.focus_block_logical_id = focus_block_logical_id
        self.show_voronoi_boundaries = show_voronoi_boundaries
        self.max_fiducials_per_district = max_fiducials_per_district

        self.focus_metrics = focus_metrics
        self.focus_metric_width = focus_metric_width
        self.focus_metric_height = focus_metric_height
        self.screen_size = screen_size
        self.n_focus_rows = rows = int(screen_size[1] // focus_metric_height)
        self.n_focus_cols = cols = int(math.ceil(len(focus_metrics) / rows))
        self.focus_region_width = cols * focus_metric_width

        if not self.table_mode:
            h = 34 * len(self.district_blocks_fid) + 5 * (
                len(self.district_blocks_fid) - 1)
            box = self.gui_touch_focus_buttons = BoxLayout(
                orientation='vertical', size=('100dp', '{}dp'.format(h)),
                spacing='5dp', pos=(self.focus_region_width, 0))

            for val in self.district_blocks_fid:
                btn = ToggleButton(
                    text='District {}'.format(val), group='focus',
                    allow_no_selection=False)
                box.add_widget(btn)

                def update_current_fid(*largs, button=btn, value=val):
                    if button.state == 'down':
                        self.current_fid_id = value
                btn.fbind('state', update_current_fid)

            btn = ToggleButton(
                text='Focus', group='focus', allow_no_selection=False)
            box.add_widget(btn)

            def update_current_fid(*largs, button=btn):
                if button.state == 'down':
                    self.current_fid_id = self.focus_block_logical_id
            btn.fbind('state', update_current_fid)

            box.children[-1].state = 'down'
            self.add_widget(box)

        i = 0
        for col in range(cols):
            for row in range(rows):
                name = focus_metrics[i]
                x0 = col * focus_metric_width
                x1 = x0 + focus_metric_width
                y0 = row * focus_metric_height
                y1 = y0 + focus_metric_height

                self.add_widget(Factory.SizedLabel(text=name, pos=(x0, y0)))
                with self.canvas:
                    Line(points=[x0, y0, x1, y0, x1, y1, x0, y1], width=2)

                i += 1
                if i >= len(focus_metrics):
                    break
            if i >= len(focus_metrics):
                break

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
            PushMatrix()
            Translate(self.focus_region_width, 0)
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

                graphics.append(Color(rgba=(0, 1, 0, 1)))
                graphics.append(Line(points=precinct.boundary, width=1))
                precinct_graphics[precinct] = graphics
            PopMatrix()

    def on_touch_down(self, touch):
        if not self.table_mode:
            if self.gui_touch_focus_buttons.collide_point(*touch.pos):
                return self.gui_touch_focus_buttons.on_touch_down(touch)
            return self.gui_touch_down(touch)
        return self.fiducial_down(touch)

    def on_touch_move(self, touch):
        if not self.table_mode and \
                self.gui_touch_focus_buttons.collide_point(*touch.pos):
            return self.gui_touch_focus_buttons.on_touch_down(touch)

        if touch.uid not in self.touches:
            return False

        if self.table_mode:
            return self.fiducial_move(touch)
        return self.gui_touch_move(touch)

    def on_touch_up(self, touch):
        if not self.table_mode and \
                self.gui_touch_focus_buttons.collide_point(*touch.pos):
            return self.gui_touch_focus_buttons.on_touch_down(touch)

        if touch.uid not in self.touches:
            return False

        if self.table_mode:
            return self.fiducial_up(touch)
        return self.gui_touch_up(touch)

    def align_touch(self, pos):
        if self.align_mat is not None:
            pos = tuple(
                np.dot(self.align_mat, np.array([pos[0], pos[1], 1]))[:2])

        x0, y0 = self.screen_offset
        pos = pos[0] - x0, pos[1] - y0
        return pos

    def handle_focus_block(self, pos):
        assert self.focus_metrics
        if self.ros_bridge is None:
            return

        x, y = pos
        if x < self.focus_region_width:
            rows = self.n_focus_rows

            metric = ''
            if y < rows * self.focus_metric_height:
                row = int(y / self.focus_metric_height)
                col = int(x / self.focus_metric_width)
                metric = self.focus_metrics[col * rows + row]
            self.ros_bridge.update_tuio_focus(False, metric)
        else:
            district = self.voronoi_mapping.get_pos_district(
                (x - self.focus_region_width, y))

            # it's not on any district, send a no block present signal
            if district is None:
                self.ros_bridge.update_tuio_focus(False, '')
            else:
                self.ros_bridge.update_tuio_focus(True, district.identity)

    def focus_block_down(self, touch, pos):
        # there's already a focus block on the table
        if self._has_focus or not self.focus_metrics:
            return True
        self._has_focus = touch

        with self.canvas:
            color = Color(rgba=(1, 0, 1, 1))
            point = Point(points=pos, pointsize=7)

        info = {'fid': touch.fid, 'last_pos': pos, 'graphics': (color, point),
                'focus': True}
        self.touches[touch.uid] = info

        self.handle_focus_block(pos)
        return True

    def focus_block_move(self, touch, pos):
        """Only called in table mode and if the touch has been seen before
        and it is a focus block.
        """
        info = self.touches[touch.uid]
        info['last_pos'] = pos
        info['graphics'][1].points = pos

        self.handle_focus_block(pos)
        return True

    def focus_block_up(self, touch):
        """Only called in table mode and if the touch has been seen before
        and it is a focus block.
        """
        info = self.touches[touch.uid]
        for item in info['graphics']:
            self.canvas.remove(item)

        del self.touches[touch.uid]
        self._has_focus = None

        if self.ros_bridge is not None:
            self.ros_bridge.update_tuio_focus(False, '')
        return True

    def fiducial_down(self, touch):
        focus_id = self.focus_block_logical_id
        blocks_fid = self.district_blocks_fid
        if 'markerid' not in touch.profile or (
                touch.fid not in blocks_fid and touch.fid != focus_id):
            return False

        x, y = pos = self.align_touch(touch.pos)

        # handle focus block
        if touch.fid == focus_id:
            return self.focus_block_down(touch, pos)
        if x < self.focus_region_width:
            return True

        with self.canvas:
            color = Color(rgba=(1, 1, 1, 1))
            point = Point(points=pos, pointsize=7)

        logical_id = blocks_fid.index(touch.fid)
        key = self.add_fiducial((x - self.focus_region_width, y), logical_id)

        info = {'fid': touch.fid, 'fiducial_key': key, 'last_pos': pos,
                'graphics': (color, point), 'logical_id': logical_id}
        self.touches[touch.uid] = info

        self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def fiducial_move(self, touch):
        """Only called in table mode and if the touch has been seen before.
        """
        info = self.touches[touch.uid]
        x, y = pos = self.align_touch(touch.pos)
        if info['last_pos'] == pos:
            return True

        if 'focus' in info:
            return self.focus_block_move(touch, pos)

        info['last_pos'] = pos
        info['graphics'][1].points = pos

        self.voronoi_mapping.move_fiducial(
            info['fiducial_key'], (x - self.focus_region_width, y))
        self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def fiducial_up(self, touch):
        """Only called in table mode and if the touch has been seen before.
        """
        info = self.touches[touch.uid]
        if 'focus' in info:
            return self.focus_block_up(touch)

        del self.touches[touch.uid]
        for item in info['graphics']:
            self.canvas.remove(item)

        x, y = self.align_touch(touch.pos)
        self.remove_fiducial(
            info['fiducial_key'], (x - self.focus_region_width, y))
        self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def gui_touch_down(self, touch):
        x, y = pos = self.align_touch(touch.pos)
        info = {'moved': False, 'fiducial_key': None}

        # are we near a voronoi touch?
        x_offset = self.focus_region_width
        for key, (x2, y2) in self.voronoi_mapping.get_fiducials().items():
            if ((x - x_offset - x2) ** 2 + (y - y2) ** 2) ** .5 < 10:
                info['fiducial_key'] = key
                self.touches[touch.uid] = info
                return True

        # are we near the focus touch?
        if self.focus_gui_pos:
            x2, y2 = self.focus_gui_pos
            if ((x - x2) ** 2 + (y - y2) ** 2) ** .5 < 10:
                info['focus'] = True
                self.touches[touch.uid] = info
                return True

        # handle focus down
        if self.current_fid_id is self.focus_block_logical_id:
            if self.focus_gui_pos or not self.focus_metrics:
                return True
            self.focus_gui_pos = pos

            with self.canvas:
                color = Color(rgba=(1, 0, 1, 1))
                point = Point(points=pos, pointsize=7)
            self.fiducial_graphics['focus'] = color, point
            info['focus'] = True
            info['moved'] = True

            self.touches[touch.uid] = info
            self.handle_focus_block(pos)
            return True

        if x < self.focus_region_width:
            return True

        # with self.canvas:
        #     color = Color(rgba=(1, 1, 1, 1))
        #     point = Point(points=pos, pointsize=7)

        current_id = self.current_fid_id
        if len(
                [1 for val in self.voronoi_mapping.get_fiducial_ids().values()
                 if val == current_id]) >= self.max_fiducials_per_district:
            return True

        key = self.add_fiducial(
            (x - self.focus_region_width, y), current_id)

        label = self.fiducial_graphics[key] = Label(
            text=str(self.current_fid_id), center=tuple(map(float, pos)),
            font_size='20sp')
        self.add_widget(label)
        info['fiducial_key'] = key
        info['moved'] = True
        self.touches[touch.uid] = info

        self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def gui_touch_move(self, touch):
        """Only called when not in table mode and if the touch has been seen
        before.
        """
        x, y = pos = self.align_touch(touch.pos)
        info = self.touches[touch.uid]
        info['moved'] = True

        if 'focus' in info:
            if self.focus_gui_pos != pos:
                self.handle_focus_block(pos)
            self.focus_gui_pos = self.fiducial_graphics['focus'][1].points = pos
            return True

        key = info['fiducial_key']
        pos_ = (x - self.focus_region_width, y)
        if self.voronoi_mapping.get_fiducials()[key] != pos_:
            self.fiducial_graphics[key].center = tuple(map(float, pos))
            self.voronoi_mapping.move_fiducial(key, pos_)
            self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def gui_touch_up(self, touch):
        """Only called when not in table mode and if the touch has been seen
        before.
        """
        x, y = pos = self.align_touch(touch.pos)

        info = self.touches.pop(touch.uid)
        if 'focus' in info:
            # if moved, we leave point on gui
            if info['moved']:
                if self.focus_gui_pos != pos:
                    self.handle_focus_block(pos)
                self.focus_gui_pos = self.fiducial_graphics['focus'][1].points = pos
                return True
            # if it didn't move, we remove the point
            for item in self.fiducial_graphics['focus']:
                self.canvas.remove(item)
            del self.fiducial_graphics['focus']

            self.focus_gui_pos = None
            if self.ros_bridge is not None:
                self.ros_bridge.update_tuio_focus(False, '')
            return True

        key = info['fiducial_key']
        pos_ = (x - self.focus_region_width, y)
        if info['moved']:
            if self.voronoi_mapping.get_fiducials()[key] != pos_:
                self.fiducial_graphics[key].center = tuple(map(float, pos))
                self.voronoi_mapping.move_fiducial(key, pos_)
                self.voronoi_mapping.request_reassignment(self.voronoi_callback)
            return True

        self.remove_widget(self.fiducial_graphics[key])
        # for item in self.fiducial_graphics[key]:
        #     self.canvas.remove(item)
        del self.fiducial_graphics[key]

        self.remove_fiducial(key, pos_)
        self.voronoi_mapping.request_reassignment(self.voronoi_callback)
        return True

    def add_fiducial(self, location, identity):
        fiducial = self.voronoi_mapping.add_fiducial(location, identity)
        if identity not in self.fiducials_color:
            self.fiducials_color[identity] = list(next(self.colors))
        return fiducial

    def remove_fiducial(self, fiducial, location):
        self.voronoi_mapping.remove_fiducial(fiducial)

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
            self, districts, fiducial_identity, fiducial_pos, error=[],
            post_callback=None, largs=(),
            data_is_old=False):
        if data_is_old:
            return

        if post_callback is not None:
            post_callback(*largs)

        if not error:
            fid_ids = [self.district_blocks_fid[i] for i in fiducial_identity]
            if self.ros_bridge is not None:
                self.ros_bridge.update_voronoi(
                    fiducial_pos, fid_ids, fiducial_identity, districts,
                    self.district_metrics_fn, self.state_metrics_fn)
            if not districts:
                self.clear_voronoi()
                return

        colors = self.fiducials_color
        for district in districts:
            color = colors[district.identity]
            for precinct in district.precincts:
                self.precinct_graphics[precinct][0].rgba = color + [1., ]

        if error:
            for precinct in error:
                self.precinct_graphics[precinct][0].a = .3

        for item in self.district_graphics:
            self.canvas.remove(item)
        self.district_graphics = []

        if self.show_voronoi_boundaries:
            PushMatrix()
            Translate(self.focus_region_width, 0)
            with self.canvas:
                self.district_graphics.append(Color(1, 1, 0, 1))
                for district in districts:
                    self.district_graphics.append(
                        Line(points=district.boundary + district.boundary[:2],
                             width=2))
            PopMatrix()


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

    ros_host = 'localhost'

    ros_port = 9090

    show_voronoi_boundaries = False

    focus_metrics = []

    focus_metric_width = 100

    focus_metric_height = 100

    metric_data = None

    log_data = False

    max_fiducials_per_district = 5

    def load_metrics(self):
        assert self.use_county_dataset
        names = {'Saint Croix': 'St. Croix'}
        root = os.path.join(
            os.path.dirname(distopia.__file__), 'data', 'aggregate')

        self.metric_data = MetricData(
            root_path=root, metrics=self.metrics, precinct_names_map=names,
            precincts=self.precincts)

    def load_precinct_adjacency(self):
        assert self.use_county_dataset
        fname = os.path.join(
            os.path.dirname(distopia.__file__), 'data', 'county_adjacency.json')

        with open(fname, 'r') as fh:
            counties = json.load(fh)

        precincts = self.precincts
        for i, neighbours in counties.items():
            precincts[int(i)].neighbours = [precincts[p] for p in neighbours]

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
        vor.start_processing_thread()
        vor.screen_size = self.screen_size
        self.precincts = precincts = []

        for i, (record, polygons) in enumerate(
                zip(geo_data.records, geo_data.polygons)):
            precinct = Precinct(
                name=record[3], boundary=polygons[0].reshape(-1).tolist(),
                identity=i, location=polygons[0].mean(axis=0).tolist())
            precincts.append(precinct)

        vor.set_precincts(precincts)

    def show_precinct_labels(self, widget):
        offset = widget.focus_region_width
        for i, precinct in enumerate(self.precincts):
            x, y = precinct.location
            x += offset
            label = Label(
                text=str(precinct.identity), center=(x, y),
                font_size=20)
            widget.add_widget(label)

    def load_config(self):
        keys = ['use_county_dataset', 'screen_size',
                'table_mode', 'alignment_filename', 'screen_offset',
                'show_precinct_id', 'focus_block_fid',
                'focus_block_logical_id', 'district_blocks_fid', 'use_ros',
                'metrics', 'ros_host', 'ros_port', 'show_voronoi_boundaries',
                'focus_metrics', 'focus_metric_width', 'focus_metric_height',
                'log_data', 'max_fiducials_per_district']

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

        for metric in self.focus_metrics:
            if metric not in self.metrics:
                raise ValueError(
                    'Cannot enable focus metric "{}" because it\'s not in '
                    'metrics "{}"'.format(metric, self.metrics))

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
        self.load_metrics()
        self.load_precinct_adjacency()

        widget = VoronoiWidget(
            voronoi_mapping=self.voronoi_mapping,
            table_mode=self.table_mode, align_mat=mat,
            screen_offset=self.screen_offset, ros_bridge=self.ros_bridge,
            district_blocks_fid=self.district_blocks_fid,
            focus_block_fid=self.focus_block_fid,
            focus_block_logical_id=self.focus_block_logical_id,
            district_metrics_fn=self.metric_data.compute_district_metrics,
            state_metrics_fn=self.metric_data.create_state_metrics,
            show_voronoi_boundaries=self.show_voronoi_boundaries,
            focus_metrics=self.focus_metrics, screen_size=self.screen_size,
            focus_metric_height=self.focus_metric_height,
            focus_metric_width=self.focus_metric_width,
            max_fiducials_per_district=self.max_fiducials_per_district)

        if self.use_ros:
            box = BoxLayout()
            voronoi_widget = widget
            err = Label(text='No ROS bridge. Please set use_ros to False')
            widget = box
            box.add_widget(err)

            def enable_widget():
                box.remove_widget(err)
                box.add_widget(voronoi_widget)
                voronoi_widget.ros_bridge = self.ros_bridge
                if self.show_precinct_id:
                    self.show_precinct_labels(voronoi_widget)

            self.ros_bridge = RosBridge(
                host=self.ros_host, port=self.ros_port,
                ready_callback=enable_widget, log_data=self.log_data)
        else:
            if self.show_precinct_id:
                self.show_precinct_labels(widget)
        self._profiler = widget._profiler = cProfile.Profile()
        return widget


Builder.load_string("""
<SizedLabel@Label>:
    size: self.texture_size
""")

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
