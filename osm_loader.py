import os
import requests

def request_big_data(url, osm_file):
    req = requests.get(url,stream=True)
    with open(osm_file, 'wb') as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    data_path = 'datasets'
    city = 'manhattan'
    if city == 'manhattan':
        bbox = [-74.028, 40.695, -73.940, 40.788]
    else:
        bbox = [-80.035, 40.425, -79.930, 40.460]
    osm_file = os.path.join(data_path, city, (city +'.osm'))  

    if not os.path.isfile(osm_file):
        url = ('http://overpass.openstreetmap.ru/cgi/xapi_meta?*[bbox='
                + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ']')
        print(url)
        request_big_data(url, osm_file)
