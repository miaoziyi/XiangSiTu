1.运行train.py：
-t:训练集文件路径
----------------------------------------------------------------------------------------------------
python '.\metric learning train.py' --train_csv ../../txt/per_leafCid_250_1_StratifiedKFold_5.csv

2.运行infer.py：
-m：训练好的模型路径
-t:测试集文件路径
-i:测试集图片路径
------------------------------
测试集格式：
imgPath
XX.jpg
XX.jpg
XX.jpg
-----------------------------
python '.\metric learning infer.py' -m D:\mzy\mzy\main\model\model_efficientnet_b3_IMG_SIZE_512_arcface.bin -t D:\mzy\mzy\main\txt\img_test_path_new_rename.csv -i D:/xiangsitu/img_test_new_rename/

3.运行find.py：
-m：训练好的模型路径
-tc：测试集文件路径
-ta：搜索集绝对路径
-tr：测试集相对路径
-qc：搜索集文件路径
-te：测试集嵌入向量路径
-----------------------------
测试集格式：
imgPath
XX.jpg
XX.jpg
XX.jpg
---------------------------------------
搜索集格式：
imgPath
绝对路径/XX.jpg
绝对路径/XX.jpg
绝对路径/XX.jpg
如D:\xiangsitu\img_test_new_rename/30_26_556.jpg
----------------------------------
搜索集绝对路径和测试集相对路径：
由于 测试图片目录：D:\xiangsitu\img_test_new_rename/
-ta：搜索集绝对路径填写D:\xiangsitu\
-tr：测试集相对路径填写img_test_new_rename/
保证最后生成的json文件图片路径均为img_test_new_rename/XX.jpg
----------------------------------------------------------------------
python '.\metric learning find.py' -m D:\mzy\mzy\main\model\metric250\newTest\model_efficientnet_b3_IMG_SIZE_512_arcface_250_5.bin -tc D:\mzy\mzy\main\txt\query_test\img_test_path_new_rename.csv -ta D:/xiangsitu/ -tr img
_test_new_rename/ -qc D:\mzy\mzy\main\txt\query_test\query-imgPath_new.csv -te D:\mzy\mzy\main\data\metric250\newTest\metric_learning_250_5.npy
