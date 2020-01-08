from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path,
                                   "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
for i in range(8):
    imagenr = 'images/IMG_222{}.jpg'
    imagenr = imagenr.format(i)
    output_imagenr = 'images/output/IMG_222{}.jpg'
    output_imagenr = output_imagenr.format(i)
    detections = detector.detectObjectsFromImage(input_image=imagenr,
                                                 output_image_path=
                                                 output_imagenr)

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])
