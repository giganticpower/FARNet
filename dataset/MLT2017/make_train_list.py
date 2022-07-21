import os

os.environ["CUDA_VISIBLE_DEVICE"] = "3"

data_root = "/data/chh/DRRG-master/data/MLT2017/"
images_path = data_root + "ch4_test_images"

images_list = os.listdir(images_path)

file = open(data_root + "test_list", "w")

for i in images_list:
    file.write(i)
    file.write("\n")
file.close()