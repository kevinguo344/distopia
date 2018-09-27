"""
Census and Election Data
============================

Some of our data sources:

https://elections.wi.gov/elections-voting/statistics
https://github.com/nvkelso/election-geodata/issues/2
https://doi.org/10.7910/DVN/NH5S2I
https://hub.arcgis.com/datasets/f0532acbe3304e20a11dbd598c5654f7_0
https://elections.wi.gov/elections-voting/results/2016/fall-general
https://elections.wi.gov/sites/default/files/11.4.14%20Election%20Results-all%20offices-w%20x%20w%20report.pdf
https://www.cityofmadison.com/planning/unit_planning/map_aldermanic/ald_dist_8x10.pdf
https://data-wi-dnr.opendata.arcgis.com/datasets/county-boundaries-24k

We assume that in wisconsin, wards are precincts.
"""

import os.path
import distopia
import shapefile
import numpy as np
from pyproj import Proj, transform
import math

__all__ = ('GeoData', )


class GeoData(object):
    """Represents the census data and the polygons associated with the data.
    """

    records = None

    shapes = None

    polygons = None

    fields = None

    shp_reader = None

    source_coordinates = 'epsg:3071'

    target_coordinates = 'epsg:3857'

    dataset_name = 'WI_Election_Data_with_2017_Wards'

    screen_size = (1920, 1080)

    containing_rect = None
    """4-tuple describing the rectangle containing all the polygons.
    
    ``(min_x, min_y, max_x, max_y)``
    """

    def load_data(self):
        """Loads the data from file.
        """
        data_path = os.path.join(
            os.path.dirname(distopia.__file__), 'data',
            self.dataset_name, self.dataset_name)

        self.shp_reader = shp = shapefile.Reader(data_path)
        self.fields = fields = [
            f[0] for f in shp.fields if not isinstance(f, tuple)]
        self.shapes = list(shp.shapes())
        self.records = records = list(shp.records())

        assert len(records)
        assert len(records[0]) == len(fields)

    def generate_polygons(self):
        """Converts the shapes in the data into a list of closed polygons,
        described by their vertices.
        """
        src, target = self.source_coordinates, self.target_coordinates
        if not src or not target:
            trans = lambda _0, _1, lon, lat: (lon, lat)
            crs_src = crs_target = None
        else:
            trans = transform
            crs_src = Proj(init='epsg:3071')
            crs_target = Proj(init='epsg:3857')

        self.polygons = polygons = []
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for record, shape in zip(self.records, self.shapes):
            if len(shape.parts) == 1:
                arr = np.array(
                    [trans(crs_src, crs_target, lon, lat) for
                     lon, lat in shape.points])

                max_x = max(max_x, np.max(arr[:, 0]))
                max_y = max(max_y, np.max(arr[:, 1]))
                min_x = min(min_x, np.min(arr[:, 0]))
                min_y = min(min_y, np.min(arr[:, 1]))

                poly_list = [arr, ]
            else:
                poly_list = []
                for i, s in enumerate(shape.parts):
                    if i < len(shape.parts) - 1:
                        e = shape.parts[i + 1] - 1
                    else:
                        e = len(shape.points)

                    arr = np.array([
                        trans(crs_src, crs_target, lon, lat) for
                        lon, lat in shape.points[s:e + 1]])

                    max_x = max(max_x, np.max(arr[:, 0]))
                    max_y = max(max_y, np.max(arr[:, 1]))
                    min_x = min(min_x, np.min(arr[:, 0]))
                    min_y = min(min_y, np.min(arr[:, 1]))

                    poly_list.append(arr)
            polygons.append(poly_list)
        self.containing_rect = min_x, min_y, max_x, max_y

    def scale_to_screen(self):
        """Scales the polygons to the screen size.
        """
        if self.screen_size is None:
            raise TypeError('A screen size was not given')

        min_x, min_y, max_x, max_y = self.containing_rect
        width, height = map(float, self.screen_size)
        # normalize to the screen size
        max_x, max_y = math.ceil(max_x), math.ceil(max_y)
        min_x, min_y = math.floor(min_x), math.floor(min_y)
        x_ratio = width / (max_x - min_x)
        y_ratio = height / (max_y - min_y)
        ratio = min(x_ratio, y_ratio)

        for precinct_polygons in self.polygons:
            for polygon in precinct_polygons:
                polygon[:, 0] -= min_x
                polygon[:, 0] *= ratio
                polygon[:, 1] -= min_y
                polygon[:, 1] *= ratio
