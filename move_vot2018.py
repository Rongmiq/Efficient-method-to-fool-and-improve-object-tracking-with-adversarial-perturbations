import os
import shutil
#
data_root = os.path.join('/home/xmq/Desktop/dataset/VOT2018/')
color_root = os.path.join('/home/xmq/Desktop/dataset/VOT2018/ball2/color')
# shutil.move(data_root,color_root)


data = []
for line in open("/home/xmq/Desktop/dataset/VOT2018/list.txt","r"):
    data.append(line.replace('\n','/'))

for i in data[3:]:
    dir = os.path.join(data_root,i)
    for f in os.listdir(dir):
        if f.endswith('.jpg'):
            # print(f)
            src = os.path.join(dir,f)
            dst = os.path.join(dir,'color/')
            shutil.move(src, dst)