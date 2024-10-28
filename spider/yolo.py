import json
import torch

class YOLOInference:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        if torch.cuda.is_available():
            self.model.to('cuda')
    def inference(self, img_path: str) -> dict:
        """
        经过YOLO推理返回json。
        json['xmin'][id: str], json['ymin'][id: str]: float, 物体id的左上角坐标
        json['xmax'][id: str], json['ymax'][id: str]: float, 物体id的右下角坐标
        json['confidence'][id: str]: float, 置信度
        json['name'][id: str]: str, 类别名称
        """
        results = self.model(img_path)
        json_str = results.pandas().xyxy[0].to_json()
        return json.loads(json_str)

if __name__ == '__main__':
    inference = YOLOInference()
    result = inference.inference('data/1.jpeg')
    print(result)