
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

    parser.add_argument(
        'layer_index',
        type=int,
        help='Index of layer for which to to find activations.'
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
input_img_shape = np.array(list(model.layers[0].input_shape[1:3])).T
output = model.layers[args.layer_index].output
functor = K.function([input]+ [K.learning_phase()], [output])


# In[ ]:

base_path = image_folder 
if not image_folder[-1] == '/':
    base_path += '/'

out_dir_path = base_path+'respones/'+model_name.split('-')[0] +'/'
  
os.makedirs(out_dir_path,exist_ok=True)
print("Saving activation maps to: ",out_dir_path)


for filename in tqdm(os.listdir(image_folder)):
    if not os.path.isfile(image_folder+'/'+filename):
        print('Skipping ',filename)
        continue

    img = cv2.cvtColor(cv2.imread(image_folder+'/'+filename),cv2.COLOR_BGR2RGB)

    if not np.array_equal(img.shape[0:2],input_img_shape):
        img = cv2.resize(img,tuple(input_img_shape))
        img = np.swapaxes(img, 0, 1)

    activations = functor([img[None,:,:,:],False])
    
    maps = activations[0]
    maps = (maps.squeeze() + 0.5)* 255
    maps = maps.astype(np.uint8)
   

        
    for i in range(maps.shape[-1]):
        img_out = maps[:,:,i]
        prefix,ext = filename.split('.')
        cv2.imwrite(out_dir_path+prefix+'_map-'+str(i)+'.'+ext,img_out)
