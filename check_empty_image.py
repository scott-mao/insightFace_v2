import os
import tqdm
import multiprocessing
from glob import glob
import shutil

f = open('list.txt', 'w')
images = os.listdir('/home/tupm/Documents/augmented')

# with open('4401561.jpg', 'rb')  as fp:
#     print(len(fp.read()))

def action(e):
    path = os.path.join('/home/tupm/Documents/augmented', e)
    with open(path, 'rb')  as fp:
        if len(fp.read()) == 0:
            # print(e)
            f.write(e+'\n')
            with open('/home/tupm/HDD/datasets/2d_face/insight/face/faces_emore/data/augmented/' + e, 'rb') as f2:
                leng = len(f2.read())
                
                if leng != 0:
                    shutil.copy('/home/tupm/HDD/datasets/2d_face/insight/face/faces_emore/data/augmented/' + e, path)
                else:
                    print('augmented_', e, leng)
                    with open('/home/tupm/HDD/datasets/2d_face/insight/face/faces_emore/data/images/' + e, 'rb') as f2:
                        leng = len(f2.read())
                        
                        if leng != 0:
                            shutil.copy('/home/tupm/HDD/datasets/2d_face/insight/face/faces_emore/data/images/' + e, path)
                        else:
                            print('images: ', e,  leng)
                            f.write(e+'\n')

for _ in tqdm.tqdm(multiprocessing.Pool(12).imap_unordered(action, images)):
    _