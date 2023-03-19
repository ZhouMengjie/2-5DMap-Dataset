import os
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
    parser.add_argument('--dataroot', type=str, required=True, default='datasets', help='dataset root folder')
    parser.add_argument('--city', type=str, required=True, default='manhattan', help='city name')
    args = parser.parse_args()

    data_path = args.dataroot
    city = args.city
    if city == 'manhattan':
        bbox = [-74.028, 40.695, -73.940, 40.788]
    elif city == 'pittsburgh':
        bbox = [-80.035, 40.425, -79.930, 40.460]
    else:
        raise NotImplementedError('Please manually set the bounding box!')
    
    folder_path = os.path.join(data_path, city)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    assert os.path.exists(folder_path), 'Cannot create folder: {}'.format(folder_path)

    osm_file = os.path.join(folder_path, (city +'.osm'))  
    if not os.path.isfile(osm_file):
        url = ('http://overpass.openstreetmap.ru/cgi/xapi_meta?*[bbox='
                + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ']')
        print(url)
        request_big_data(url, osm_file)
