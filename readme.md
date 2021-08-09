# Brain Tumor Detector
## descriptions
这是一个在线检测脑瘤的应用，只要上传脑部MRI图片，就可以诊断出对应的脑肿瘤。我们可以检测出三种类型的脑肿瘤，包括glioma_tumor meningioma_tumor pituitary_tumor，当然检测结果可能为无肿瘤，如果你上传一张正常人脑MRI图片。
这个应用使用Vue.js作为前端开发的框架，提供user-friendly的用户界面，上传模块使用的是filepond插件，使用Flask作为后端，用于预测的模型运行在后端，后端主要的工作是接收前端发来的图片，用模型进行预测，最后返回结果到前端。
## quick start
我强烈建议使用kaggle或者google colab运行demo，因为这样不需要在本地配置环境，而且使用统一的环境，可以避免安装环境时出现错误。如果使用kaggle或者google colab运行demo，可以使用这个[notebook](./brain_tumor_mri_classification_tensorflow.ipynb).

以下是本地安装的方法。
```cmd
pip3 install requirements.txt
python3 main.py
```

## model train tutorial
如果你想从头开始训练自己的脑肿瘤检测模型，我推荐你看[notebook](./my_medical_app_demo.ipynb),里面包含了数据准备，数据预处理，模型训练，模型预测等end-to-end的详细过程。

## todo
- [ ]提高模型的精度，现在是98%，是否可以提升到99%甚至更高？
- [ ]对更多种类的模型进行训练
- [ ]完善UI界面
