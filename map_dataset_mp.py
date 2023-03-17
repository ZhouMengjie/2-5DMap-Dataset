import os
import sys
sys.path.append(os.getcwd())
import tqdm
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from openstreetmap import cropping

class RenderThread:
    def __init__(self, points_mh,points_pt,radius,save_path,q,printLock):
        self.points_mh = points_mh
        self.points_pt = points_pt
        self.radius = radius
        self.save_path = save_path
        self.q = q 
        self.printLock = printLock

    def loop(self):
        global counter
        while True:
            info = self.q.get()
            if (info.any() == None):
                self.q.task_done()
                break
            else:
                panoid = info[0]
                center_xy = [info[1], info[2]]
                city = info[4]

            if os.path.exists(os.path.join(self.save_path, (panoid + '.npy'))):
                print(panoid+' exists...')
            else:
                # print(panoid+' processing...')
                # just crop
                if city == 'manhattan':
                    points = self.points_mh
                elif city == 'pittsburgh':
                    points = self.points_pt
                else:
                    center_xy = []
                    R = []
                    assert len(center_xy)==0 or len(R) == 0, f"Cannot find correspondent points in both datasets!"

                vertex_indices = cropping.crop_pcd_v2(center_xy, points, self.radius)
                if len(vertex_indices) > 0:
                    np.save(os.path.join(self.save_path, (panoid + '.npy')), vertex_indices)

            self.printLock.acquire()
            self.printLock.release()
            self.q.task_done()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate map datasets')
    parser.add_argument('--area', type=str, required=False, default='hudsonriver5kU', help='map dataset root folder')
    parser.add_argument('--radius', type=np.int64, required=False, default=114, help='map dataset root folder')    
    parser.add_argument('--num_threads', type=np.int64, required=False, default=16, help='map dataset root folder')        
    args = parser.parse_args()

    data_path = os.path.join(os.getcwd(), 'datasets')
    # load pcd
    points_mh = np.load(os.path.join(data_path, 'manhattan', 'manhattanU.npy'))
    points_pt = np.load(os.path.join(data_path, 'pittsburgh', 'pittsburghU.npy'))                    

    # read csv file
    area = args.area

    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    infos = data.values
    radius = args.radius

    # Create threads
    # print(multiprocessing.cpu_count())
    # num_threads = multiprocessing.cpu_count() # 8
    num_threads = args.num_threads
    queue = multiprocessing.JoinableQueue(40)
    printLock = multiprocessing.Lock()
    renderers = {}
    if not os.path.isdir(os.path.join(data_path, (area+'_idx'))):
        os.makedirs(os.path.join(data_path, (area+'_idx')))

    for i in range(num_threads):
        renderer = RenderThread(points_mh,points_pt,radius,os.path.join(data_path, (area+'_idx')),queue,printLock)
        render_thread = multiprocessing.Process(target=renderer.loop)
        render_thread.start()
        renderers[i] = render_thread

    for i in tqdm.tqdm(range(infos.shape[0])):
        t = (infos[i])
        queue.put(t)


    print("No more locations ...")
    # Signal render threads to exit by sending empty request to queue
    for i in range(num_threads):
        queue.put(np.asarray(None))

    # wait for pending rendering jobs to complete
    queue.join()

    for i in range(num_threads):
        renderers[i].join()




            







    
        
        

            






