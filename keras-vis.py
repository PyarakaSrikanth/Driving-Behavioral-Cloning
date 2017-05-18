
# coding: utf-8

# In[ ]:
import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
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
    

# Load model.
model = load_model(model_name)



# Get input and output layers and input layer size.
# Input layer sie will tell us if the images for
# which activation needs to be generated have to be
# resized.
#
# The size computed here is nrowsxncols not widthxheight.

input = model.input
input_img_shape = np.array(list(model.layers[0].input_shape[1:3]))
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

    # OpenCV arranges channels in BGR order while model uses RGB.
    # Switch channel order.
    img = cv2.cvtColor(cv2.imread(image_folder+'/'+filename),cv2.COLOR_BGR2RGB)

    # Resize (if needed). Destination size is ncolsxnrows.
    img = cv2.resize(img,(input_img_shape[1],input_img_shape[0]))


    # Call the underlying function to generate output on this image.
    activations = functor([img[None,:,:,:],False])

    # Since there's only one image the activations list has only one member.
    # Get first member from the list. This gives us all output layers.
    maps = activations[0]
    maps = (maps.squeeze() + 0.5)* 255
    maps = maps.astype(np.uint8)
   
    # Combine all activation maps. We take max to highlight any pixel
    # that is active in any of the maps. The last axes is the index
    # of the map layer.
    combined_map = np.amax(maps,axis=2)
    print(combined_map.shape)
    prefix, ext = filename.split('.')
    cv2.imwrite(out_dir_path + prefix + '-layer' + str(args.layer_index) + '-map.'
                + ext, combined_map)
