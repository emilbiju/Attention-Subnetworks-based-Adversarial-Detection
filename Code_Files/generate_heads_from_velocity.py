import numpy as np
import os
import tensorflow as tf

num_hidden_layers = 12
num_heads_per_layer = 12

def get_ckpt_num_list(output_dir):
	'''Extract checkpoint numbers from ckpt file names'''
	ckpt_num_list = []
	for filename in os.listdir(output_dir):
		if filename.startswith('model.ckpt-') and filename.endswith('.meta'):
			ckpt_num_list.append(int(filename[11:-5]))
	ckpt_num_list.sort()
	print(ckpt_num_list)
	return ckpt_num_list

def get_prun_masks_all_ckpts(ckpt_num_list, output_dir):
    '''Get the trained pruning mask values across all checkpoints.
    Returns numpy array of size (num_checkpoints, num_hidden_layers, num_heads_per_layer)'''

    prun_mask_all_ckpts = []
    for ckpt_num in ckpt_num_list:
        reader = tf.train.NewCheckpointReader(os.path.join(output_dir,'model.ckpt-'+str(ckpt_num)))
        shapes_dict = reader.get_variable_to_shape_map()  # get the variable names
        prun_mask = []
        for layer in range(num_hidden_layers):
            extracted_values_prun = reader.get_tensor('bert/encoder/layer_'+str(layer)+'/attention/self/pruning_mask_layer_name')
            extracted_values_prun = list(np.reshape(extracted_values_prun, num_heads_per_layer))
            prun_mask.append(extracted_values_prun)
        prun_mask_all_ckpts.append(prun_mask)
        
    prun_mask_all_ckpts = np.array(prun_mask_all_ckpts)
    return prun_mask_all_ckpts

def get_prun_mask_to_use(output_dir):
    '''Returns a pruning mask created using a threshold defined based on
    the difference between initial and final values of each mask.'''
    
    ckpt_num_list = get_ckpt_num_list(output_dir)
    prun_mask_all_ckpts = get_prun_masks_all_ckpts(ckpt_num_list, output_dir)
    diff = prun_mask_all_ckpts[-1] - prun_mask_all_ckpts[0]
    thresh = np.max(diff)/2
    prun_mask_to_use = (diff>=thresh).astype(int).tolist()
    np.save(os.path.join(output_dir, 'prun_masks_all_ckpts'),prun_mask_all_ckpts)
    np.save(os.path.join(output_dir, 'prun_mask_to_use'),prun_mask_to_use)  
    return prun_mask_all_ckpts, prun_mask_to_use
