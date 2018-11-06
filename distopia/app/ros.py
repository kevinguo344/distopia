"""
ROS Bridge
=============

Publishes the state of the system to ROS.
"""

from threading import Thread
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import json
import roslibpy
import logging
import datetime
import os
import distopia

__all__ = ('RosBridge', )


class RosBridge(object):
    """Maintains subscribers and publishers to a ROS master via rosbridge
    """

    _publisher_thread = None

    _publisher_thread_queue = None

    ros = None

    ready_callback = None

    log_data = False

    def __init__(self, host="localhost", port=9090, ready_callback=None,
                 log_data=False):
        self.ready_callback = ready_callback
        self.log_data = log_data

        self.ros = ros = roslibpy.Ros(host=host, port=port)
        ros.on_ready(self.start_publisher_thread)
        ros.connect()

    def start_publisher_thread(self):
        logging.info('Connected to ros-bridge')

        designs_topic = roslibpy.Topic(
            self.ros, '/evaluated_designs', 'std_msgs/String')
        designs_topic.advertise()

        blocks_topic = roslibpy.Topic(
            self.ros, '/blocks', 'std_msgs/String')
        blocks_topic.advertise()

        tuio_topic = roslibpy.Topic(
            self.ros, '/tuio_control', 'std_msgs/String')
        tuio_topic.advertise()
        logging.info('Started ros-bridge publishers')

        self._publisher_thread_queue = Queue()
        self._publisher_thread = thread = Thread(
            target=self.publisher_thread_function,
            args=(designs_topic, blocks_topic, tuio_topic))
        thread.start()

        if self.ready_callback:
            self.ready_callback()

    def stop_threads(self):
        if self._publisher_thread is not None:
            self._publisher_thread_queue.put(('eof', None))
            self._publisher_thread.join()

    def update_tuio_focus(self, focus_district, focus_param):
        self._publisher_thread_queue.put(
            ('focus', (focus_district, focus_param)))

    def update_voronoi(
            self, fiducials_locations, fiducial_ids, fiducial_logical_ids,
            districts, district_metrics_fn, state_metrics_fn):
        self._publisher_thread_queue.put(
            ('voronoi',
             (fiducials_locations, fiducial_ids, fiducial_logical_ids,
              districts, district_metrics_fn, state_metrics_fn)))

    @staticmethod
    def make_computation_packet(
            fiducials_locations, fiducial_ids, fiducial_logical_ids,
            districts, district_metrics_fn, state_metrics_fn):
        district_metrics_fn(districts)
        state_data = [m.get_data() for m in state_metrics_fn(districts)]

        districts_data = []
        for district in districts:
            district_data = {
                'district_id': district.identity,
                'precincts': [p.identity for p in district.precincts],
                'metrics': [m.get_data() for m in district.metrics.values()]
            }
            districts_data.append(district_data)

        blocks_data = []
        for (x, y), fid_id, logical_id in zip(
                fiducials_locations, fiducial_ids,
                fiducial_logical_ids):
            item = {
                'x': x, 'y': y, 'fid_id': fid_id,
                'logical_id': logical_id}
            blocks_data.append(item)

        return state_data, districts_data, blocks_data

    def create_log_file(self):
        root = os.path.join(os.path.dirname(distopia.__file__), 'logs')
        if not os.path.exists(root):
            os.mkdir(root)

        t = '{}'.format(datetime.datetime.utcnow()).replace(':', '.')
        fname = os.path.join(root, 'distopia_log_{}.json'.format(t))
        i = 1

        while os.path.exists(fname):
            fname = os.path.join(root, 'distopia_log_{}-{}.json'.format(t, i))
            i += 1
        return open(fname, 'w')

    def publisher_thread_function(
            self, designs_topic, blocks_topic, tuio_topic):
        assert self.ros is not None
        queue = self._publisher_thread_queue
        packet_count = 0

        fh = None
        first = True
        if self.log_data:
            fh = self.create_log_file()
            fh.write('[')

        try:
            while True:
                item, val = queue.get(block=True)
                if item == 'eof':
                    tuio_topic.unadvertise()
                    blocks_topic.unadvertise()
                    designs_topic.unadvertise()
                    return

                if item == 'focus':
                    focus_district, focus_param = val
                    cmd = 'focus_district' if focus_district else 'focus_state'
                    param = str(focus_param)

                    obj = {'cmd': cmd, 'param': param}
                    tuio_topic.publish({'data': json.dumps(obj)})

                    log_obj = {
                        'focus': obj,
                        'utc_time': '{}'.format(datetime.datetime.utcnow())
                    }
                elif item == 'voronoi':
                    state_data, districts_data, blocks_data = \
                        self.make_computation_packet(*val)

                    count = packet_count
                    packet_count += 1

                    districts_obj = {
                        'count': count, 'districts': districts_data,
                        'metrics': state_data}
                    designs_topic.publish({'data': json.dumps(districts_obj)})

                    fiducials_obj = {'count': count, 'fiducials': blocks_data}
                    blocks_topic.publish({'data': json.dumps(fiducials_obj)})

                    log_obj = {
                        'districts': districts_obj,
                        'fiducials': fiducials_obj,
                        'utc_time': '{}'.format(datetime.datetime.utcnow())
                    }
                else:
                    assert False

                if fh is not None:
                    if first:
                        first = False
                    else:
                        fh.write(',\n')
                    json.dump(log_obj, fh)
        finally:
            if fh is not None:
                fh.write('\n]')
                fh.close()
