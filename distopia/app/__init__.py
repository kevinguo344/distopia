"""Distopia Application
========================

Runs the application.
"""


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from distopia.app.geo_data import GeoData


def mpl_plot_data(geo_data):
    colors = cycle(['r', 'g', 'b', 'y'])
    for record, polygons in zip(geo_data.records, geo_data.polygons):
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


if __name__ == '__main__':
    from distopia.app.geo_data import GeoData
    geo_data = GeoData()
    geo_data.dataset_name = 'WI_Election_Data_with_2017_Wards'
    geo_data.load_data()
    geo_data.generate_polygons()
    mpl_plot_data(geo_data)
