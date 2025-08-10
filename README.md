# Архитектура
В качестве детектора использовал YoloV11N. Сконверитровал ее в ONNX. По bbox считал опорные точки по сетке. Затем при помощи calcOpticalFlowPyrLK вычислял оптисечкий поток по двум изображениям и базовым точкам полученным с детектора.

# Примечания 
До конца не уверен что правильно собран sln файл. Собирал автоматически. Разрабатывал из vscode на linux. Есть два варианта на текущей версии под visual studio и без ide (ветка linux) Использовал opencv-4.9

# Результаты:
![image](result/flow_0000.png)

![image](result/flow_0001.png)

![image](result/flow_0002.png)

![image](result/flow_0003.png)

![image](result/flow_0004.png)

![image](result/flow_0005.png)

![image](result/flow_0006.png)

![image](result/flow_0007.png)

![image](result/flow_0008.png)

![image](result/flow_0009.png)