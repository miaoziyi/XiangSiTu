import os

path = r"D:\mzy\test"
filelist = os.listdir(path)  # 返回该目录下所有文件及文件夹的名字，并存放于一个列表中
# print(filelist)

txt_tables = []
f = open("txt/predict.txt", "r", encoding='utf-8')
line = f.readline()  # 读取第一行
while line:
    txt_data = eval(line)  # 可将字符串变为元组
    txt_tables.append(txt_data)  # 列表增加
    line = f.readline()  # 读取下一行


count = 0
for file in filelist:
    Old_dir = os.path.join(path, file)  # 连接路径与文件名

    if os.path.isdir(Old_dir):  # 如果不是图片而是一个文件夹目录，那么不重新命名，跳过对下一条文件名操作
        continue
    filename = os.path.splitext(file)[0]  # 将图片的文件名分离成文件名与拓展名，如：'01.jpg'→元组('01','.jpg')
    filetype = os.path.splitext(file)[1]
    New_dir = os.path.join(path, filename+'_'+str(txt_tables[count]).zfill(1) + filetype)
    os.rename(Old_dir, New_dir)  # 重命名
    count += 1
