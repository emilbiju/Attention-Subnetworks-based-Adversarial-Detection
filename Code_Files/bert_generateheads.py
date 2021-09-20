import numpy as np
import sys
import random
from string import Template

def sample_heads(total_heads,sampled_heads):
	try:
		samples = random.sample(range(0,total_heads), sampled_heads)  # random sampling without replacement
		return samples
	except ValueError:
		print("Sample size exceeded total size")

def generate_binary_array(total_heads,sampled_heads):
	binary_array = np.zeros(total_heads, dtype=int)
	for i in sampled_heads:
		binary_array[i] = 1
	return binary_array

def generate_head_string(template_string,head_array):
	t = Template(template_string)
	temp_dict = {}
	for i in range(len(head_array)):
		temp = "h{}".format(i)
		temp_dict[temp] = head_array[i]
	head_string = t.substitute(temp_dict)
	return head_string


# BERT : Encoder Self Attention
n = int(sys.argv[1])
mask_string = 'pruning_mask = "[[$h0,$h1,$h2,$h3,$h4,$h5,$h6,$h7,$h8,$h9,$h10,$h11],\
[$h12,$h13,$h14,$h15,$h16,$h17,$h18,$h19,$h20,$h21,$h22,$h23],\
[$h24,$h25,$h26,$h27,$h28,$h29,$h30,$h31,$h32,$h33,$h34,$h35],\
[$h36,$h37,$h38,$h39,$h40,$h41,$h42,$h43,$h44,$h45,$h46,$h47],\
[$h48,$h49,$h50,$h51,$h52,$h53,$h54,$h55,$h56,$h57,$h58,$h59],\
[$h60,$h61,$h62,$h63,$h64,$h65,$h66,$h67,$h68,$h69,$h70,$h71],\
[$h72,$h73,$h74,$h75,$h76,$h77,$h78,$h79,$h80,$h81,$h82,$h83],\
[$h84,$h85,$h86,$h87,$h88,$h89,$h90,$h91,$h92,$h93,$h94,$h95],\
[$h96,$h97,$h98,$h99,$h100,$h101,$h102,$h103,$h104,$h105,$h106,$h107],\
[$h108,$h109,$h110,$h111,$h112,$h113,$h114,$h115,$h116,$h117,$h118,$h119],\
[$h120,$h121,$h122,$h123,$h124,$h125,$h126,$h127,$h128,$h129,$h130,$h131],\
[$h132,$h133,$h134,$h135,$h136,$h137,$h138,$h139,$h140,$h141,$h142,$h143]]"'

total_heads = 144
layer_heads = 12
mask_samples = sample_heads(total_heads,n)
print(mask_samples)
mask_array = generate_binary_array(total_heads,mask_samples)
enc = generate_head_string(mask_string,mask_array)
print(enc)
