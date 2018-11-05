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

__all__ = ('RosBridge', )


class RosBridge(object):
    """Maintains subscribers and publishers to a ROS master via rosbridge
    """

    _publisher_thread = None

    _publisher_thread_queue = None

    ros = None

    ready_callback = None

    def __init__(self, host="localhost", port=9090, ready_callback=None):
        self.ready_callback = ready_callback

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

    def publisher_thread_function(
            self, designs_topic, blocks_topic, tuio_topic):
        assert self.ros is not None
        queue = self._publisher_thread_queue
        packet_count = 0

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
                encoded_str = json.dumps({'cmd': cmd, 'param': param})

                tuio_topic.publish({'data': encoded_str})
            elif item == 'voronoi':
                state_data, districts_data, blocks_data = \
                    self.make_computation_packet(*val)

                count = packet_count
                packet_count += 1

                encoded_str = json.dumps(
                    {'count': count, 'districts': districts_data,
                     'metrics': state_data})
                designs_topic.publish({'data': encoded_str})

                encoded_str = json.dumps(
                    {'count': count, 'fiducials': blocks_data})
                blocks_topic.publish({'data': encoded_str})
