import csv
import numpy as np
with open("add_results_real.csv","w") as f:
		f.write("num_layer,num_param,d_loss_min,d_loss_final,g_loss_min,g_loss_final\n")
for i in range(1,76):
	reader = csv.DictReader(open('datas/add_results_'+str(i)+'.txt'))
	min_g_loss=1000
	min_d_loss=1000
	g_loss_last=0
	d_loss_last=0
	for row in reader:
		if np.float64(row['g_loss'])<min_g_loss:
			min_g_loss=np.float64(row['g_loss'])
		if np.float64(row['d_loss'])<min_d_loss:
			min_d_loss=np.float64(row['d_loss'])
	cnt=0
	g_loss_last=np.float64(row['g_loss'])
	d_loss_last=np.float64(row['d_loss'])
	with open('datas/add_report'+str(i)+".txt") as f:
		for row in f:
			cnt+=1
			if "Trainable" in row:
				x=str(row)
	x=int(x.replace("Trainable params: ","").replace(",",""))
	num_layer=cnt-15
	with open("add_results_real.csv","a") as f:
		f.write(str(num_layer)+","+str(x)+","+str(min_d_loss)+","+str(d_loss_last)+","+str(min_g_loss)+","+str(g_loss_last)+","+"\n")