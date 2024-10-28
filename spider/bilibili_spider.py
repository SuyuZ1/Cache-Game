import os
import json
import time
import yaml
import logging
import requests
from bvid import BilibiliScraper
from spider_utils import VideoProcessor

class BilibiliSpider:
    def __init__(self, cfg_path: str='config/spider.yaml'):
        """
        :param cfg_path: 配置文件路径
        """
        with open(cfg_path, 'r', encoding='utf-8') as yaml_file:
            self.cfg = yaml.safe_load(yaml_file)
        self.history_path = self.cfg['history_file']
        if not os.path.exists(self.history_path):
            with open(self.history_path, 'w') as f:
                self.history = {'last_page': 1,
                                '1': []}
                json.dump(self.history, f)
                f.close()
        else:
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
                f.close()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(self.cfg['log_path'])
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        
    def get_cid(self, bv_id: str) -> str:
        """
        B站视频内部用cid表示，获取弹幕需要从BV号转cid.
        :param bv_id: 视频BV号
        :return: 视频的cid
        """
        url = f'https://api.bilibili.com/x/player/pagelist?bvid={bv_id}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        req = requests.get(url, headers=headers)
        result = req.json()
        assert 'data' in result # result['data'][0]['cid']

        return result['data'][0]['cid']
    def get_danmaku(self, bv_id: str, xml_path: str) -> None:
        """
        将指定BV号的弹幕保存为xml文件。
        :param bv_id: BV号
        :param xml_path: xml文档保存路径
        """
        cid = self.get_cid(bv_id)
        danmaku_url = f'https://comment.bilibili.com/{cid}.xml'
        req = requests.get(danmaku_url)
        req.encoding = 'utf-8'
        if req.status_code == 200:
            with open(xml_path, 'wb') as file:
                file.write(req.content)
                file.close()
    
    def get_video(self, bv_id: str, video_path: str) -> None:
        """
        将指定BV号的弹幕保存为视频文件。由于B站视频和音频不是一起存的，所以视频只有画面没有声音。
        :param bv_id: BV号
        :param video_path: 视频保存路径
        """
        url = f'https://www.bilibili.com/video/{bv_id}'
        headers = {
            'referer':'https://www.bilibili.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'
        }
        req = requests.get(url, headers=headers)
        req.encoding = 'utf-8'
        
        begin_token = '"baseUrl":"'
        end_token = '"'
        text = req.text
        start_pos = text.find(begin_token) + len(begin_token)
        end_pos = text.find(end_token, start_pos)
        if start_pos == -1:
            raise Exception('未获取视频源地址!')
        video_url = text[start_pos: end_pos]
        video_request = requests.get(video_url, headers=headers)
        
        with open(video_path, 'wb') as f:
            f.write(video_request.content)
            f.close()
    
    def save_history(self) -> None:
        """
        保存历史记录。
        """
        with open(self.cfg['history_file'], 'w') as f:
            json.dump(self.history, f)
            f.close()
    
    def run(self, debug: bool=False) -> None:
        """
        进行数据爬取。
        :param debug: 是否显示中间结果
        """
        last_page = self.history['last_page']
        scraper = BilibiliScraper()
        cnt = 0
        videoes = []
        for page_num in range(last_page, 0, -1):
            videoes += self.history[str(page_num)]
        videoes = set(videoes)
            
        processor = VideoProcessor(self.cfg['csv_path'],
                                   self.cfg['scoring_threshold'],
                                   self.cfg['confidence_threshold'],
                                   self.cfg['sample_rate'],
                                   self.cfg['k_neighbor'],
                                   self.cfg['clean_enabled'])
        while True:
            logging.info(f'Start from page {last_page}')
            bvids = scraper.get_page(last_page)
            logging.info(f'Get page {last_page} bvids({len(bvids)})')
            for bvid in bvids:
                if bvid in videoes:
                    logging.info(f'Downloaded {bvid}, skip')
                    continue
                # hadn't downloaded
                if not bvid in self.history[str(last_page)]:
                    self.get_danmaku(bvid, f'data/xml/{bvid}.xml')
                    logging.info(f'Successfully downloaded danmaku file of {bvid}')

                    self.get_video(bvid, f'data/video/{bvid}.mp4')
                    logging.info(f'Successfully downloaded video file of {bvid}')

                    size = processor.process(f'data/video/{bvid}.mp4', f'data/xml/{bvid}.xml', debug=debug)
                    logging.info(f'Append {size} items to csv file')
                    
                    cnt += 1
                    self.history[str(last_page)].append(bvid)
                    # save history file
                    if cnt == self.cfg['limit']:
                        self.save_history()
                        return
                    if cnt % self.cfg['save_interval'] == 0:
                        self.save_history()
                    
                    if self.cfg['interval']:
                        time.sleep(self.cfg['interval'])
            last_page += 1
            self.history[str(last_page)] = []
            self.history['last_page'] = last_page

if __name__ == '__main__':
    spider = BilibiliSpider('config/spider.yaml')
    spider.run()