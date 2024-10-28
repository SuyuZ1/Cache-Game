import os
import cv2
import numpy as np
from tqdm import tqdm
import PySimpleGUI as sg
from collections import Counter
from yolo import YOLOInference
from typing import List, Tuple
from pose import MMPoseWrapper
import xml.etree.ElementTree as ET
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def get_time_from_attr(attributes: str) -> float:
    """
    从p属性获取该弹幕于视频的开始时间。
    :param attributes: 从parse中获取的p属性attribute字符串
    :return: 返回该弹幕的开始时间(s)。
    """
    attrs = attributes.split(',')
    start_time = attrs[0]
    return float(start_time)

def parse_xml(xml_path: str, sort=True) -> List[Tuple[float, str]]:
    """
    将指定xml文件转为表示弹幕的列表。列表的每个元素是一个二元组(start_time: float, text: str),
    分别表示开始时间(秒)和文本。
    :param xml_path: xml文档路径
    :param sort: 是否按时间排序
    :return: List[Tuple[float, str]], 如上文所述
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = []
    for child in root:
        if 'p' in child.attrib:
            attributes = child.get('p')
            text = child.text
            result.append((get_time_from_attr(attributes), text))
    if sort:
        result.sort()
    return result

class VideoProcessor:
    """
    该类用于从视频中提取监督信息，用于后续训练。
    """
    def __init__(self, csv_path, scoring_threshold=5, confidence_threshold=0.7, sample_rate=5, k_neighbor=10, clean_enabled=-1):
        """
        :param csv_path: 用于存储数据的csv文件
        :param scoring_threshold: 只有某帧附近的弹幕数 > scoring_threshold 才会被考虑
        :param confidence_threshold: 只有置信度 > confidence_threshold 才会被考虑
        :param sample_rate: 采样率，每 sample_rate 帧采集一次数据
        :param clean_enabled: 若 clean_enabled < 0, 不清理data文件夹；否则每调用 `VideoProcessor.process` `clean_enabled`次会
        自动清理`data`文件夹，防止消耗过多磁盘空间。
        """
        self.current_ptr = 0
        self.yolo_inf = YOLOInference()
        self.pose_inf = MMPoseWrapper()
        self.scoring_threshold = scoring_threshold
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.csv_path = csv_path
        self.process_cnt = 0
        self.k_neighbor = k_neighbor
        self.clean_enabled = clean_enabled
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.close()
        
        np.set_printoptions(threshold=10000)
    
    def _get_neighbor_danmaku(self, danmaku: List[Tuple[float, str]], time_stamp: float, k_neighbor=10) -> List[str]:
        """
        从弹幕列表中找出附近的弹幕。选取规则: 选取弹幕发送时间 >= time_stamp 的至多k_neighbor条弹幕。
        :param danmaku: parse_xml返回的弹幕列表, **需要按照时间排序。**
        :param time_stamp: 指定的时间戳
        :param k_neighbor: 至多选取多少条弹幕
        :return: List[str], 选取的弹幕列表
        """
        while self.current_ptr < len(danmaku) and danmaku[self.current_ptr][0] < time_stamp:
            self.current_ptr += 1
        tuples = danmaku[self.current_ptr: self.current_ptr+k_neighbor]
        danmakus = [item[1] for item in tuples]
        return danmakus

    def _get_danmaku_score(self, danmakus: List[str]) -> float:
        """
        给定弹幕列表，返回该弹幕的(情感)得分。该得分会作为该帧的监督信号。
        目前得分会在所有弹幕上进行平均。
        :param danmakus: List[str], 弹幕列表
        :return: 弹幕情感得分score, 0 <= score <= 1
        """
        if not hasattr(self, 'score_model'):
            self.score_model = pipeline(Tasks.text_classification, 'damo/nlp_structbert_sentiment-classification_chinese-large')
        
        result = self.score_model(danmakus)
        result = np.array([item['scores'][int(item['labels'][0] != '正面')] for item in result]) # 0不是'正面'对应下标就是1
        return float(np.mean(result))

    def is_interested_frame(self, frame: np.ndarray) -> Tuple[bool, dict]:
        """
        给定帧，确定该帧是否包含感兴趣内容(比如检测到猫)。只有感兴趣内容才会被进一步处理(如提取关节)。
        :param frame: np.ndarray(h, w), uint8表示的帧
        :return: Tuple[bool, dict], 该帧是否包含感兴趣内容。若包含感兴趣内容，返回(True, 识别结果json); 否则返回(False, None).
        """
        frame = frame[..., ::-1]
        results = self.yolo_inf.inference(frame)
        counter = Counter(results['name'].values())
        if counter.get('cat', 0) == 1 and results['confidence']['0'] > self.confidence_threshold:
            return True, results
        return False, None
    
    def get_feature(self, frame: np.ndarray, show: bool=False) -> Tuple[bool, np.ndarray]:
        """
        给定帧，从该帧中提取关节特征信息。若提取失败，返回(False, None).
        :param frame: np.ndarray(h, w), uint8表示的帧
        :return: (bool, np.ndarray), 从该帧中提取得到的关节信息，用于构建数据集进行训练。
        """
        try:
            inf_result = self.pose_inf.inference(frame, show=show)
        except:
            return False, None
        if len(inf_result['predictions'][0]) != 1:
            return False, None
        key_points = inf_result['predictions'][0][0]['keypoints']
        # 归一化
        key_points = np.array(key_points)
        key_points[:, 0] /= frame.shape[1] # 第 0 列是x(横), 除以shape[1]
        key_points[:, 1] /= frame.shape[0] # 第 1 列是y(纵), 除以shape[0]

        return True, key_points

    def process(self, video_path: str, xml_path: str, debug=False) -> int:
        """
        从视频中提取训练数据，保存到csv文件中。
        :param video_path: 视频路径
        :param xml_path: 弹幕路径
        :param debug: 查看中间结果
        :return: int, 从该视频中提取得到的数据组数
        """
        cap = cv2.VideoCapture(video_path)
        danmaku = parse_xml(xml_path, sort=True)
        data = []
        self.current_ptr = 0
        batch_size = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in tqdm(range(total_frames), desc='Processing frames...', unit='frame'):
            try:
                ret, frame = cap.read()

                if frame_num % self.sample_rate != 0:
                    continue

                time_stamp = cap.get(cv2.CAP_PROP_POS_MSEC) # 该帧的时间戳(ms)
                time_stamp /= 1000 # ms -> s

                current_danmaku = self._get_neighbor_danmaku(danmaku, time_stamp, self.k_neighbor)

                if len(current_danmaku) > self.scoring_threshold: # 弹幕数量足够多
                    interested, result = self.is_interested_frame(frame) 
                    if interested: # 该帧为感兴趣帧(比如包含猫的图像)
                        # x: 横, y: 纵
                        cat_id = [key for key, value in result['name'].items() if value == 'cat'][0]
                        x_min, y_min = int(result['xmin'][cat_id]), int(result['ymin'][cat_id])
                        x_max, y_max = int(result['xmax'][cat_id]), int(result['ymax'][cat_id])
                        roi = frame[y_min:y_max, x_min:x_max,:]

                        danmaku_score = self._get_danmaku_score(current_danmaku)
                        flag, features = self.get_feature(roi, show=debug)
                        if debug:
                            cv2.imshow(f'cat at {time_stamp}', roi)
                            sg.popup(f'{current_danmaku}, score={danmaku_score}', title=f'cat at {time_stamp}')
                            cv2.destroyAllWindows()

                        # (features, danmaku_score) 为一组数据, 0 <= danmaku_score <= 1是监督信号
                        if flag:
                            features = features.flatten()
                            data.append(np.concatenate([features, np.array([danmaku_score])]))
                            batch_size += 1
            except:
                continue

        cap.release()
        # append to csv
        if len(data) == 0: # no interested frame
            return 0
        
        data = np.stack(data)
        data_str = np.array2string(data, separator=',', max_line_width=10000).replace('[', '').replace(']', '').replace(' ', '').replace(',\n', '\n')
        with open(self.csv_path, 'a') as f:
            f.write(data_str + '\n')
            f.close()
        
        self.process_cnt += 1
        if self.clean_enabled > 0 and self.process_cnt % self.clean_enabled == 0:
            os.system('rm data/video/*')
            os.system('rm data/xml/*')

        return batch_size

if __name__ == '__main__':
    processor = VideoProcessor(csv_path='data/train.csv')
    processor.process('data/video/BV1hz4y1c7oQ.mp4', 'data/xml/BV1hz4y1c7oQ.xml', debug=True)