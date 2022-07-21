import os
import scipy.io as scio
import cv2
import itertools
import re
import numpy as np


synth_path = '/data/chh/DRRG-master/data/SynthText/SynthText'

synthtext_folder = synth_path
gt = scio.loadmat(os.path.join(synthtext_folder, 'gt.mat'))
print('loading gt access')
wordbox = gt['wordBB'][0]
image = gt['imnames'][0]
imgtxt = gt['txt'][0]
annotation_root = os.path.join(synthtext_folder,'gt')

for index in range(len(image)):
        img_path = os.path.join(synthtext_folder, str(image[index][0]))
        image_name = image[index][0]
        #img_id = image[index][0].split('/')[1]
        #img_id_txt = img_id.replace('.jpg', '.txt')
        #word_annotation_path = os.path.join(annotation_root, img_id_txt)
        #image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(wordbox[index].shape) == 2:
                wordboxs = wordbox[index].transpose((1, 0)).reshape(-1, 8)
        else:
                wordboxs = wordbox[index].transpose((2, 1, 0)).reshape(-1, 8)
        words = [re.split(' \n|\n |\n| ', t.strip()) for t in imgtxt[index]]
        words = list(itertools.chain(*words))
        with open('train.txt', 'a+') as f:
                f.write(image_name + '\t')
                f.close()
        for i in range(len(wordboxs)):
                word = words[i]
                wordboxes = wordboxs[i]
                with open('train.txt', 'a+') as f:
                        dict1 = {"transcription": word}
                        dict2 = {"points": wordboxes}
                        if i == 0:
                                f.write('[')
                        else:
                                f.write(', ')
                        f.write('{')
                        for key in dict1:
                                f.writelines('"' + str(key) + '": ' + '"' + str(dict1[key]) + '"')
                        f.write(', ')
                        for key1 in dict2:
                                f.writelines('"' + str(key1) + '": ')
                                f.writelines('[')
                                f.writelines('[' + str(dict2[key1][0]) + ', ' + str(dict2[key1][1]) + '], ')
                                f.writelines('[' + str(dict2[key1][2]) + ', ' + str(dict2[key1][3]) + '], ')
                                f.writelines('[' + str(dict2[key1][4]) + ', ' + str(dict2[key1][5]) + '], ')
                                f.writelines('[' + str(dict2[key1][6]) + ', ' + str(dict2[key1][7]) + ']')
                                f.writelines(']')
                        f.write('}')
                        f.close()
        with open('train.txt', 'a+') as f:
                f.write(']')
                f.write('\n')
                f.close()
        print(index)
print('compelated')
