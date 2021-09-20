# This script runs fine tuning on BERT for the mentioned number of epochs in the train sh file.
# Then, evaluates on all the checkpoints stored.
# Then, it choses the best checkpoint out of all, i.e. the one giving the maximum accuracy.
# Then, it predicts on the best checkpoint.

# For example: For qqp task, this script needs 3 files in the same directory: qqp_train.sh, qqp_eval.sh and qqp_predict.sh
# which are replicas of run_bert.sh with do_train=true, do_eval=true and do_predict=true respectively.

from string import Template
import collections
import operator
import glob as g
import re
import os

task = 'qqp'
pp = 'zero_percentage_pruning'
log_dir = './logs/' + task + '/'
output_dir = '/home/newdir/' + pp + '/' + task + '_dir/'
gs_output_dir = 'gs://my_bucket/' + pp + '/' + task + '_dir/'
checkpoint_file = str(output_dir+'checkpoint')

# ====== TRAIN ===== #
cmd = "bash " + task + "_train.sh " + log_dir + 'train_log ' + gs_output_dir
os.system(cmd)

# ====== EVAL ====== #
def get_checkpoint_string(x):
	t = Template('model_checkpoint_path: "model.ckpt-$ckpt"\n')
	temp_dict = {}
	temp_dict['ckpt'] = str(x)
	final_string = t.substitute(temp_dict)
	return final_string

all_ckpts = []
with open(checkpoint_file, 'r') as file:
	all_ckpts = file.readlines()
	all_ckpts.pop(0)
	for i in range(len(all_ckpts)):
		all_ckpts[i] = re.findall('\d+', all_ckpts[i])[0]
print(all_ckpts)

for i in all_ckpts:
	lines = []
	with open(checkpoint_file, 'r') as file:
		lines = file.readlines()
		lines[0] = get_checkpoint_string(i)

	with open(checkpoint_file, 'w') as file:
		file.writelines(lines)

	cmd = "bash " + task + "_eval.sh " + log_dir + i + "_" + task + " " + gs_output_dir
	print(cmd)
	os.system(cmd)


# ===== Multiple eval is done. Now process the logs ===== #
cmd = log_dir + '/*' + task + '*'
log_files = (g.glob(cmd))
gs_eval_acc_dict = {}
e_loss = {}
t_loss = {}
epochs = {}

for i in log_files:
	with open(i, 'r') as file:
		lines = file.read()
		ckpt_num = int(re.findall('\d+',i)[0])
		ckpt_eval_acc = re.findall('\d+\.\d+', re.findall('  eval_accuracy = \d+\.\d+',lines)[0])[0]
		ckpt_eval_loss = re.findall('\d+\.\d+', re.findall('  eval_loss = \d+\.\d+',lines)[0])[0]
		ckpt_train_loss = re.findall('\d+\.\d+', re.findall('  loss = \d+\.\d+',lines)[0])[0]
		gs_eval_acc_dict[ckpt_num] = ckpt_eval_acc
		e_loss[ckpt_num] = ckpt_eval_loss
		t_loss[ckpt_num] = ckpt_train_loss

e_acc = collections.OrderedDict(sorted(gs_eval_acc_dict.items()))

k = 1
for i in e_acc:
	epochs[i] = k
	k += 1

ckpt = max(e_acc.iteritems(),key=operator.itemgetter(1))[0]
acc = e_acc[ckpt]
epoch = epochs[ckpt]
eloss = e_loss[ckpt]
tloss = t_loss[ckpt]

for k,v in epochs.iteritems():
	print(str(k) + " : " + str(v))


results_file = log_dir + 'results'
with open(results_file,'w') as file:
	file.write("epoch : eval_accuracy : eval_loss : train_loss\n")
	for k,v in e_acc.iteritems():
		file.write(str(epochs[k]) + " : ")
		file.write(str(k) + " : ")
		file.write(str(v) + " : ")
		file.write(str(e_loss[k]) + " : ")
		file.write(str(t_loss[k]))
		file.write("\n")
	file.write("\n*** Maximum eval accuracy details *** \n\n")
	file.write("Eval accuracy : {}\n".format(acc))
	file.write("Eval loss : {}\n".format(eloss))
	file.write("Train loss : {}\n".format(tloss))
	file.write("Global step : {}\n".format(ckpt))
	file.write("Epoch : {}\n".format(epoch))


# ===== PREDICT on max eval accuracy ckpt ===== #
lines = []
# set checkpoint file to ckpt corresponding to max_eval_accuracy
with open(checkpoint_file,'r') as file:
	lines = file.readlines()
	new_line = 'model_checkpoint_path: "model.ckpt-"' + str(ckpt) + '"\n'
	lines[0] = new_line
with open("checkpoint",'w') as file:
	file.writelines(lines)

cmd = "bash " + task + "_predict.sh " + log_dir + 'predict_log ' + gs_output_dir
os.system(cmd)
print("Script complete!")
