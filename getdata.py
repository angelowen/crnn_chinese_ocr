import json
import os,cv2

wr_dir = "./data/"
if not os.path.exists(wr_dir):
    os.makedirs(wr_dir)
json_src = "/home/jefflin/aicup/train/json/"
img_src = "/home/jefflin/aicup/train/img/"
files = os.listdir(json_src)

for name in files:
    filename = json_src+name ## open filename
    img = cv2.imread(f"{img_src}{name.split('.')[0]}.jpg")
    print(f"{img_src}{name.split('.')[0]}.jpg")
    write_name = wr_dir+name.split('.')[0]+'.txt' ## writing filename
    print(filename)
    with open(write_name, 'w') as fw:
        with open(filename,'r') as fr:
            data = json.load(fr)
            H = float(data["imageHeight"])
            W = float(data["imageWidth"])
            for idx,item in enumerate(data["shapes"]):
                fw.write(f"{name.split('.')[0]}_{idx}.jpg,{item['label']}\n")
                points_x,points_y = [],[]  
                for point in item["points"]:
                    x,y = point
                    points_x.append(x)
                    points_y.append(y)

                # xl_up ,yl_up ,xr_up ,yr_up ,xr_dn ,yr_dn ,xl_dn ,yl_dn = points
                crop_img = img[min(points_y):max(points_y), min(points_x):max(points_x)]
                cv2.imwrite(f"{wr_dir}/{name.split('.')[0]}_{idx}.jpg", crop_img)
    break