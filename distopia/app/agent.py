import os
import json
import numpy as np
import csv

import distopia
from distopia.app.geo_data import GeoData
from distopia.precinct import Precinct
from distopia.mapping.voronoi import VoronoiMapping
from distopia.precinct.metrics import PrecinctHistogram, PrecinctScalar
from distopia.district.metrics import DistrictHistogramAggregateMetric, \
    DistrictScalarAggregateMetric


class VoronoiAgent(object):

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

    ros_host = 'localhost'

    ros_port = 9090

    show_voronoi_boundaries = False

    focus_metrics = []

    focus_metric_width = 100

    focus_metric_height = 100

    data_loader = None

    def create_district_metrics(self, districts):
        for district in districts:
            for name in self.metrics:
                if name == 'income':
                    continue
                district.metrics[name] = \
                    DistrictHistogramAggregateMetric(
                        district=district, name=name)

            name = 'income'
            if name in self.metrics:
                district.metrics[name] = \
                    DistrictScalarAggregateMetric(district=district, name=name)

    def load_precinct_metrics(self):
        assert self.use_county_dataset

        geo_data = self.geo_data
        names = set(r[3] for r in geo_data.records)
        names = {v: v for v in names}
        names['Saint Croix'] = 'St. Croix'

        root = os.path.join(
            os.path.dirname(distopia.__file__), 'data', 'aggregate')
        for name in self.metrics:
            if name == 'income':
                continue

            fname = os.path.join(root, '{}.csv'.format(name))
            with open(fname) as fh:
                reader = csv.reader(fh, delimiter='\t')
                header = next(reader)[1:]

                data = {}
                for row in reader:
                    data[row[0]] = list(map(float, row[1:]))

            for precinct, record in zip(self.precincts, geo_data.records):
                name = names[record[3]]
                precinct.metrics[name] = PrecinctHistogram(
                    name=name, labels=header, data=data[name])

        name = 'income'
        if name in self.metrics:
            fname = os.path.join(root, '{}.csv'.format(name))
            with open(fname) as fh:
                reader = csv.reader(fh, delimiter='\t')
                _ = next(reader)[1:]  # header

                data = {}
                for row in reader:
                    data[row[0]] = float(row[1])

            for precinct, record in zip(self.precincts, geo_data.records):
                name = names[record[3]]
                precinct.metrics[name] = PrecinctScalar(
                    name=name, value=data[name])

    def load_precinct_adjacency(self):
        assert self.use_county_dataset
        fname = os.path.join(
            os.path.dirname(distopia.__file__), 'data', 'county_adjacency.json')

        with open(fname, 'r') as fh:
            counties = json.load(fh)

        precincts = self.precincts
        for i, neighbours in counties.items():
            precincts[int(i)].neighbours = [precincts[p] for p in neighbours]

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

    def load_config(self):
        keys = ['use_county_dataset', 'screen_size',
                'table_mode', 'alignment_filename', 'screen_offset',
                'show_precinct_id', 'focus_block_fid',
                'focus_block_logical_id', 'district_blocks_fid', 'use_ros',
                'metrics', 'ros_host', 'ros_port', 'show_voronoi_boundaries',
                'focus_metrics', 'focus_metric_width', 'focus_metric_height']

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

    def load_data(self):
        """Builds the GUI.
        """
        self.load_config()
        self.create_voronoi()
        self.load_precinct_metrics()
        self.load_precinct_adjacency()

    def compute_voronoi_metrics(self, fiducials):
        vor = self.voronoi_mapping
        for fid_id, locations in fiducials.items():
            for location in locations:
                vor.add_fiducial(location, fid_id)

        districts = vor.apply_voronoi()
        self.create_district_metrics(districts)
        state_mets = self.create_state_metrics(districts)

        state_metrics = []
        for metric in state_mets:
            metric.compute()
            state_metrics.append(metric)

        district_metrics = {}
        for district in districts:
            district.compute_metrics()
            district_metrics[district.identity] = list(district.metrics.values())

        return state_metrics, district_metrics


if __name__ == '__main__':
    agent = VoronoiAgent()
    agent.load_data()
    print('data loaded')
    print(agent.compute_voronoi_metrics(
        {0: [(251., 258.)],
         1: [(751., 257.)],
         2: [(252., 756.)],
         3: [(752., 755.)]}))
    print('dead')
