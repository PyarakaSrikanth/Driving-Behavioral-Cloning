
# coding: utf-8

# In[ ]:
import argparse
import cv2
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

from keras import backend as K
from keras.models import load_model


# In[ ]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras Visualization')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where activations will be saved.'
    )

    args = parser.parse_args()
    
    model_name = args.model
    image_folder = args.image_folder
    
    
else:
    model_name = 'nvidia-f3b3new.h5'
    image_folder = 'activation-test-data/full/'
    

model = load_model(model_name)



# In[ ]:

input = model.input
output = model.layers[3].output
functor = K.function([input], [output])


# In[ ]:

base_path = image_folder 
if not image_folder[-1] == '/':
    base_path += '/'

dir_path = base_path+'respones/'+model_name.split('-')[0] +'/'
  
os.makedirs(dir_path,exist_ok=True)
print("Saving activation maps to: ",dir_path)


for filename in tqdm(os.listdir(image_folder)):
    if not os.path.isfile(filename):
        continue

    img = cv2.cvtColor(cv2.imread(image_folder+'/'+filename),cv2.COLOR_BGR2RGB)
    activations = functor([img[None,:,:,:]])
    
    maps = activations[0]
    maps = (maps.squeeze() + 0.5)* 255
    maps = maps.astype(np.uint8)
   

        
    for i in range(maps.shape[-1]):
        img_out = maps[:,:,i]
        prefix,ext = filename.split('.')
        cv2.imwrite(dir_path+prefix+'_map-'+str(i)+'.'+ext,img_out)
