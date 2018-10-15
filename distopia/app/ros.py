"""
ROS Bridge
=============

Publishes the state of the system to ROS.
"""

from threading import Thread
from queue import Queue
import json
import roslibpy
import logging


class RosBridge(object):
    """Maintains subscribers and publishers to a ROS master via rosbridge
    """

    _publisher_thread = None

    _publisher_thread_queue = None

    ros = None

    ready_callback = None

    def __init__(self, host="daarm.ngrok.io", port=80, ready_callback=None):
        self.ready_callback = ready_callback

        self.ros = ros = roslibpy.Ros(host=host, port=port)
        ros.on_ready(self.start_publisher_thread)
        ros.connect()

    def start_publisher_thread(self):
        logging.info('Connected to ros-bridge')
        self._publisher_thread_queue = Queue()
        self._publisher_thread = thread = Thread(
            target=self.publisher_thread_function)
        thread.start()

    def stop_threads(self):
        if self._publisher_thread is not None:
            self._publisher_thread_queue.put(('eof', None))
            self._publisher_thread.join()
        elif self.ros is not None:
            self.ros.terminate()

    def update_tuio_focus(self, focus_district, focus_param):
        self._publisher_thread_queue.put(
            ('focus', (focus_district, focus_param)))

    def update_voronoi(
            self, fiducials_locations, fiducial_ids, fiducial_logical_ids,
            districts):
        self._publisher_thread_queue.put(
            ('voronoi',
             (fiducials_locations, fiducial_ids, fiducial_logical_ids,
              districts)))

    def publisher_thread_function(self):
        assert self.ros is not None
        queue = self._publisher_thread_queue

        packet_count = 0
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

        if self.ready_callback:
            self.ready_callback()

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
                fiducials_locations, fiducial_ids, fiducial_logical_ids, \
                    districts = val
                count = packet_count
                packet_count += 1

                districts_data = []
                for district in districts:
                    district_data = {
                        'district_id': district.identity,
                        'precincts': [p.identity for p in district.precincts],
                        'metrics': district.compute_metrics()
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

                encoded_str = json.dumps(
                    {'count': count, 'districts': districts_data})
                designs_topic.publish({'data': encoded_str})

                encoded_str = json.dumps(
                    {'count': count, 'fiducials': blocks_data})
                blocks_topic.publish({'data': encoded_str})
