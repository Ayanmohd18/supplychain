import pandas as pd
import numpy as np

class GeoEngine:
    """
    Transforms hierarchical store data into geospatial 3D coordinates.
    Stores: CA (4), TX (3), WI (3)
    """
    def __init__(self):
        # Approximate Coordinates for M5 Walmart Stores
        self.store_coords = {
            'CA_1': {'lat': 34.0522, 'lon': -118.2437}, # LA
            'CA_2': {'lat': 34.1478, 'lon': -118.1445}, # Pasadena
            'CA_3': {'lat': 37.7749, 'lon': -122.4194}, # SF
            'CA_4': {'lat': 37.3382, 'lon': -121.8863}, # San Jose
            'TX_1': {'lat': 32.7767, 'lon': -96.7970},  # Dallas
            'TX_2': {'lat': 29.7604, 'lon': -95.3698},  # Houston
            'TX_3': {'lat': 30.2672, 'lon': -97.7431},  # Austin
            'WI_1': {'lat': 43.0389, 'lon': -87.9065},  # Milwaukee
            'WI_2': {'lat': 43.0731, 'lon': -89.4012},  # Madison
            'WI_3': {'lat': 44.5192, 'lon': -88.0198},  # Green Bay
        }

    def get_geo_data(self):
        """Generates dynamic risk and volume data for the map."""
        data = []
        np.random.seed(99)
        
        for store, coords in self.store_coords.items():
            # Volume determines height of the 3D pillar
            volume = np.random.randint(1000, 5000)
            # Risk determines color (0 = Green, 1 = Red)
            risk = np.random.uniform(0, 1)
            
            data.append({
                'store_id': store,
                'lat': coords['lat'],
                'lon': coords['lon'],
                'volume': volume,
                'risk': risk,
                'color_r': int(risk * 255),
                'color_g': int((1 - risk) * 255),
                'color_b': 0,
                'radius': 20000
            })
            
        return pd.DataFrame(data)
