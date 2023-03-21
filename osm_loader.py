import os
import sys
sys.path.append(os.getcwd())
import argparse
import requests

def request_big_data(url, osm_file):
    req = requests.get(url,stream=True)
    with open(osm_file, 'wb') as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download .osm files')
    parser.add_argument('--dataroot', type=str, required=False, default='datasets', help='dataset root folder')
    parser.add_argument('--city', type=str, required=False, default='manhattan', help='city name')
    args = parser.parse_args()

    dataroot = args.dataroot
    city = args.city
    if city == 'manhattan':
        # [minlon, minlat, maxlon, maxlat]
        bbox = [-74.028, 40.695, -73.940, 40.788] 
    elif city == 'pittsburgh':
        bbox = [-80.035, 40.425, -79.930, 40.460]
    else:
        raise NotImplementedError('Please manually set the bounding box!')
    
    data_path = os.path.join(os.getcwd(), dataroot, city)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    assert os.path.exists(data_path), 'Cannot create folder: {}'.format(data_path)

    osm_file = os.path.join(data_path, (city +'.osm'))  
    if not os.path.isfile(osm_file):
        url = ('http://overpass.openstreetmap.ru/cgi/xapi_meta?*[bbox='
                + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ']')
        print(url)
        request_big_data(url, osm_file)
