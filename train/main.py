import cv2
import yaml
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader, random_split
from network import PoseNet, KeyPointDataset
from inf_utils import VideoProcessor
from PIL import Image, ImageDraw, ImageFont

class PoseScoreWrapper:
    def __init__(self, cfg_path: str, model=None):
        """
        :param cfg_path: yaml配置文件路径
        :param model: 模型文件路径, 默认为随机初始化
        """
        with open(cfg_path, 'r', encoding='utf-8') as yaml_file:
            self.cfg = yaml.safe_load(yaml_file)
        if model:
            self.model = torch.load(model)
        else:
            self.model = PoseNet()
        
        if torch.cuda.is_available():
            self.model.to('cuda')

    def train(self, bar=False, plot=False, dist=False) -> None:
        """
        使用关键点信息进行模型训练。
        - bar: 是否显示进度条
        - plot: 是否显示loss
        - dist: 是否划分验证集合
        """
        dataset = KeyPointDataset(self.cfg['csv_path'])
        if not dist:
            train, val = random_split(dataset, [1 - self.cfg['val_ratio'], self.cfg['val_ratio']])
            val_loader = DataLoader(val, batch_size=self.cfg['batch_size'], shuffle=True)
        else:
            train = dataset
        loader = DataLoader(train, batch_size=self.cfg['batch_size'], shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        if self.cfg['criterion'] == 'MSE':
            criterion = nn.MSELoss()
        
        losses, val_losses = [], []

        for i in tqdm(range(self.cfg['epoch']), desc='Training...'):
            if bar:
                iterator = tqdm(loader, desc=f'Training epoch {i + 1}')
            else:
                iterator = loader
            self.model.train()
            for X, y in iterator:
                if torch.cuda.is_available():
                    X = X.to('cuda')
                    y = y.to('cuda')
                y = y.reshape(-1, 1)
                
                pred_y = self.model(X)
                loss = criterion(pred_y, y)
                losses.append(loss.item())
                optimizer.zero_grad()
                
                loss.backward()
                
                optimizer.step()
            
            if not dist:
                if bar:
                    iterator = tqdm(val_loader, desc=f'Evaluating epoch {i + 1}')
                else:
                    iterator = val_loader
                self.model.eval()
                for X, y in iterator:
                    if torch.cuda.is_available():
                        X = X.to('cuda')
                        y = y.to('cuda')
                    y = y.reshape(-1, 1)

                    pred_y = self.model(X)
                    loss = criterion(pred_y, y)
                    val_losses.append(loss.item())
            
            if i % self.cfg['save_epoch'] == self.cfg['save_epoch'] - 1:
                torch.save(self.model, self.cfg['model_path'] + f'/model_epoch{i + 1}.pt')
        
        if plot:
            from matplotlib import pyplot as plt
            plt.plot(losses, label='train')
            
            if not dist:
                plt.plot(val_losses, label='val')
            plt.legend()
            plt.show()
        
        torch.save(self.model, self.cfg['model_path'] + f'/model_final.pt')
    
    def inference(self, img: Union[str, np.ndarray]) -> float:
        """
        使用模型推理得分。
        :param img_path: 图片文件路径 / numpy多维数组(h, w, c)
        :return: 推理得分 0 <= score <= 1
        """
        if not hasattr(self, 'processor'):
            self.processor = VideoProcessor(self.cfg['csv_path'])
        
        if type(img) == str:
            frame = cv2.imread(img)
        else:
            frame = img
        interested, result = self.processor.is_interested_frame(frame)

        if not interested: # 不含猫/多只猫
            return 0.

        cat_id = [key for key, value in result['name'].items() if value == 'cat'][0]
        x_min, y_min = int(result['xmin'][cat_id]), int(result['ymin'][cat_id])
        x_max, y_max = int(result['xmax'][cat_id]), int(result['ymax'][cat_id])
        roi = frame[y_min:y_max, x_min:x_max,:]

        flag, features = self.processor.get_feature(roi)

        if not flag: # 无法获取关节点
            return 0.
        
        features = torch.from_numpy(features).reshape(1, -1).to(torch.float)
        if torch.cuda.is_available():
            features = features.to('cuda')
        self.model.eval()
        
        pred_score = self.model(features)
        return float(pred_score)
    
    def attach_score(self, frame: np.ndarray, score: float) -> np.ndarray:
        img = Image.fromarray(frame)

        width, height = img.size
        draw = ImageDraw.Draw(img)

        font = ImageFont.load_default().font_variant(size=30)
        text_color = (255, 0, 0)

        score_text = f"Score: {score:.2f}"

        text_length = draw.textlength(score_text, font=font)
        text_width = text_length
        x = width - text_width - 10
        y = 10

        draw.text((x, y), score_text, font=font, fill=text_color)

        return np.array(img)

    def inference_video(self, video_path: str, output_path: str, sample_rate=1) -> None:
        """
        - video_path: 视频路径
        - output_path: 推理结果路径
        - sample_rate: 采样率，每隔`sample_rate`帧处理一次
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编解码器
        video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        for frame_num in tqdm(range(total_frames), desc='Processing frames...', unit='frame'):
            ret, frame = cap.read()
            if frame is None:
                break
            if frame_num % sample_rate != 0:
                continue
            score = self.inference(frame)
            
            frame = self.attach_score(frame, score)
            video_writer.write(frame)
        
        video_writer.release()

if __name__ == '__main__':
    wrapper = PoseScoreWrapper('config/train.yaml', model='model/model_final.pt')
    wrapper.inference_video('demo/demo2.mp4', 'demo/output2.mp4', sample_rate=1)