# 安装
Python版本需要>=3.8; YOLO实现使用`https://github.com/ultralytics/yolov5`.直接安装的`torch`应该是cpu版本，没法用GPU加速。
## 1.其它部分的安装
```bash
pip install -r requirements.txt
```
## 2.MMPOSE
`MMPOSE`安装参照`https://mmpose.readthedocs.io/zh-cn/latest/installation.html`.作为Python包安装即可。
# 说明
## 数据获取
运行`spider/main.py`即可。爬虫部分的配置文件见`config/spider.yaml`.日志文件默认为`data/log.txt`.如果设备GPU可用会自动使用GPU。
```python
from bilibili_spider import BilibiliSpider

if __name__ == '__main__':
    spider = BilibiliSpider('config/spider.yaml')
    spider.run(debug=False)
```
<details>
<summary>历史</summary>

爬虫示例文件见`spider/main.py`.

`spider`文件夹主要用于爬取信息、得到监督数据。`spider.bilibili_spider.BilibiliSpider`用于从B站爬取视频以及弹幕数据。
```python
def get_danmaku(self, bv_id: str, xml_path: str) -> None:
    """
    将指定BV号的弹幕保存为xml文件。
    :param bv_id: BV号
    :param xml_path: xml文档保存路径
    """
    pass

def get_video(self, bv_id: str, video_path: str) -> None:
    """
    将指定BV号的弹幕保存为视频文件。由于B站视频和音频不是一起存的，所以视频只有画面没有声音。
    :param bv_id: BV号
    :param video_path: 视频保存路径
    """
    pass
```
`spider.spider_utils.VideoProcessor`用于从视频中提取监督信息，从而进行后续训练。

可以使用`VideoProcessor.process`获取得到的视频的数据：
```python
def process(self, video_path: str, xml_path: str, debug=False) -> int:
    """
    从视频中提取训练数据，保存(append)到csv文件中。
    :param video_path: 视频路径
    :param xml_path: 弹幕路径
    :param debug: 查看中间结果
    :return: int, 从该视频中提取得到的数据组数
    """
```
</details>

<details>
<summary>点击查看关于内部实现的说明</summary>

有三个方法是核心(TODO)：
```python
def _get_danmaku_score(self, danmakus: List[str]) -> float:
    """
    给定弹幕列表，返回该弹幕的(情感)得分。该得分会作为该帧的监督信号。
    :param danmakus: List[str], 弹幕列表
    :return: 弹幕情感得分score, 0 <= score <= 1
    """
    # TODO

def is_interested_frame(self, frame: np.ndarray) -> Tuple[bool, dict]:
    """
    给定帧，确定该帧是否包含感兴趣内容(比如检测到猫)。只有感兴趣内容才会被进一步处理(如提取关节)。
    :param frame: np.ndarray(h, w), uint8表示的帧
    :return: Tuple[bool, dict], 该帧是否包含感兴趣内容。若包含感兴趣内容，返回(True, 识别结果json); 否则返回(False, None).
    """
    # TODO

def get_feature(self, frame: np.ndarray) -> np.ndarray:
    """
    给定帧，从该帧中提取关节特征信息。
    :param frame: np.array(h, w), uint8表示的帧
    :return: np.array, 从该帧中提取得到的关节信息，用于构建数据集进行训练。
    """
    # TODO
```
实现以上三个方法之后，可以调用`VideoProcessor.process`获取标注数据对。
</details>

## 训练&推理
推理的示例代码见`train.main`.主要的类为`train.PoseScoreWrapper`;
```python
class PoseScoreWrapper:
    def __init__(self, cfg_path: str, model=None):
        """
        :param cfg_path: yaml配置文件路径
        :param model: 模型文件路径, 默认为随机初始化
        """
```
### 训练
```python
wrapper = PoseScoreWrapper('config/train.yaml')
wrapper.train()
```
### 推理
```python
wrapper = PoseScoreWrapper('config/train.yaml', 'model/inf_model.pt')
inf_result = wrapper.inference('demo/demo.jpeg') # 也可以接收(h, w, c)的np.ndarray作为输入
print(f'图片得分: {inf_result}')

wrapper.inference_video('demo/demo1.mp4', 'demo/output1.mp4', sample_rate=1)
```
### 配置文件
配置文件在`config/train.yaml`下。各参数说明见注释。