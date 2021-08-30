import os,sys
import glob
import shutil

file_path = r'D:\repos\my_wheels\file'
raw_path = r'D:\repos\my_wheels\raw'
output_path = r'D:\repos\my_wheels\output'

# 将file_path下的所有图片存入file_list
def listFunc(file_path):
    file_list = []
    for p in glob.glob(os.path.join(file_path,r'*.jpg')):
        # img_name = os.path.splitext(os.path.basename(p))
        img_name = os.path.basename(p)
        file_list.append(img_name)
    return file_list

def copyFunc(file_list, raw_path, output_path):
    for p in glob.glob(os.path.join(raw_path,r'*.jpg')):
        img_name = os.path.basename(p)
        if img_name in file_list:
            shutil.copy(os.path.join(raw_path,img_name),os.path.join(output_path, img_name))
            print('copy img: {}'.format(img_name))

def main():
    file_list = listFunc(file_path)
    copyFunc(file_list, raw_path, output_path)

if __name__ == '__main__':
    main()