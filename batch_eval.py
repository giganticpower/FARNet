import os
import subprocess

start = 0
end = 800
exp_name = "Ctw1500"
gpu_id = "1"
# tr_thresh = 0.6
# tcl_thresh = 0.5
# expend = 0.255  #MLT2017

if __name__ == "__main__":

    for epoch in range(start, end+1, 5):
        try:
            subprocess.call(['python', 'eval_TextGraph.py',
                         "--exp_name", exp_name, "--checkepoch", str(epoch), '--gpu', gpu_id])
        except:
            continue

