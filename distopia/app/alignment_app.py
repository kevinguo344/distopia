"""
Table Alignment App
========================

The app for computing the projection/fiducial alignment.

See https://pymicro.readthedocs.io/projects/pymicro/en/latest/cookbook/pointset_registration.html
for the computation.
"""

from kivy.uix.widget import Widget
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty, ObjectProperty
from kivy.clock import Clock
from kivy.graphics import Color, Line
from kivy.metrics import dp

import numpy as np
import os
import distopia
from distopia.utils import compute_affine_transform


__all__ = ('AlignmentPointWidget', 'TouchWidget', 'AlignmentApp')


class AlignmentPointWidget(Widget):
    """Displays a point to be used in alignment.
    """

    x_ratio = NumericProperty(0)
    y_ratio = NumericProperty(0)

    touch_x = NumericProperty(0)
    touch_y = NumericProperty(0)

    parent_w = NumericProperty(0)
    parent_h = NumericProperty(0)

    parent = ObjectProperty(None, allownone=True, rebind=True)


Builder.load_string("""
<AlignmentPointWidget>:
    orientation: "vertical"
    center_x: self.x_ratio * self.parent_w
    y: self.y_ratio * self.parent_h - dp(25)
    touch_x: self.x_ratio * self.parent_w
    touch_y: self.y_ratio * self.parent_h
    parent_w: self.parent.width if self.parent else 0
    parent_h: self.parent.height if self.parent else 0
    canvas:
        Line:
            circle: self.x_ratio * self.parent_w, self.y_ratio * self.parent_h, dp(20)
    Label:
        text: "Please place the\\nfiducial on the circle"
        size_hint: None, None
        size: self.texture_size
        center_x: self.center_x and self.parent.center_x
        y: self.parent.y + dp(75)
""")


class TouchWidget(Widget):
    """Root widget that manages the points.
    """

    points = [(.25, .25), (.5, .75), (.75, .25)]

    touch_pos = []

    align_widget = None

    callback_trigger = None

    last_pos = None

    last_touch = None

    trans_mat = None

    def __init__(self, **kwargs):
        super(TouchWidget, self).__init__(**kwargs)
        self.touch_pos = []
        x, y = self.points.pop(0)
        self.align_widget = AlignmentPointWidget(x_ratio=x, y_ratio=y)
        self.add_widget(self.align_widget)
        self.callback_trigger = Clock.create_trigger(self.handle_timeout, 2)

    def handle_timeout(self, *largs):
        self.touch_pos.append((
            self.align_widget.touch_x, self.align_widget.touch_y,
            self.last_pos[0], self.last_pos[1]))

        del self.last_touch.ud['alignment']
        self.last_pos = None
        self.last_touch = None

        if self.points:
            x, y = self.points.pop(0)
            self.align_widget.x_ratio = x
            self.align_widget.y_ratio = y
        else:
            self.remove_widget(self.align_widget)
            self.align_widget = None

            points = np.array(self.touch_pos)
            screen_points = points[:, 0:2]
            touch_points = points[:, 2:4]
            self.trans_mat = compute_affine_transform(
                touch_points, screen_points)

    def on_touch_down(self, touch):
        if super(TouchWidget, self).on_touch_down(touch):
            return True

        if self.align_widget is not None:
            if self.last_touch is not None:
                return True

            touch.ud['alignment'] = True
            self.last_pos = touch.pos
            self.last_touch = touch
            self.callback_trigger()
        else:
            pos = np.ones((3, ))
            pos[:2] = touch.pos
            x, y = np.dot(self.trans_mat, pos)[:2]
            with self.canvas:
                touch.ud['circle'] = Line(circle=(x, y, dp(25)))

        return True

    def on_touch_move(self, touch):
        if self.align_widget is not None:
            if 'alignment' not in touch.ud:
                return True

            x0, y0 = touch.pos
            x1, y1 = self.last_pos
            if np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) <= 1:
                return True

            self.callback_trigger.cancel()
            self.last_pos = touch.pos
            self.callback_trigger()
        else:
            if 'circle' not in touch.ud:
                return True

            pos = np.ones((3, ))
            pos[:2] = touch.pos
            x, y = np.dot(self.trans_mat, pos)[:2]
            touch.ud['circle'].circle = x, y, dp(25)
        return True

    def on_touch_up(self, touch):
        if self.align_widget is not None:
            if 'alignment' not in touch.ud:
                return True

            self.last_pos = None
            self.last_touch = None
            self.callback_trigger.cancel()
        else:
            if 'circle' not in touch.ud:
                return True
            self.canvas.remove(touch.ud['circle'])
        return True


class AlignmentApp(App):
    """Runs the alignment app.
    """

    root_widget = None

    def build(self):
        self.root_widget = root = TouchWidget()
        return root


if __name__ == "__main__":
    app = AlignmentApp()
    app.run()

    fname = os.path.join(
        os.path.dirname(distopia.__file__), 'data', 'alignment.txt')
    with open(fname, 'w') as fh:
        for touch in app.root_widget.touch_pos:
            fh.write(','.join(map(str, touch)))
            fh.write('\n')
        for row in app.root_widget.trans_mat:
            fh.write(','.join(map(str, row)))
            fh.write('\n')
