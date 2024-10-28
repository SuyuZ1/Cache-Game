import requests
from typing import List
from bs4 import BeautifulSoup

class BilibiliScraper:
    def __init__(self, keyword='猫'):
        self.keyword = keyword
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'
        }

    def get_page(self, page_id: int) -> List[str]:
        """
        得到第`page_id`页的所有bvid.
        :param page_id: page号码
        """
        url = f'https://search.bilibili.com/all?keyword={self.keyword}&page={page_id}'
        req = requests.get(url, headers=self.headers)

        text = req.text # 返回的HTML
        return self.find_links(text)

    def find_links(self, text: str) -> List[str]:
        """
        抓取link，正则化之后得到bvid
        :param text: 网页HTML代码
        :return: 
        """
        soup = BeautifulSoup(text, 'html.parser')
        links = soup.find_all('a', href=True)
        bilibili_links = [link['href'] for link in links if link['href'].startswith('//www.bilibili.com/video/')]

        # Extract BV ID from the link
        bv_ids = [link.split('/video/')[1][:-1] for link in bilibili_links]

        # Convert list to set and back to list to remove duplicates
        bv_ids = list(set(bv_ids))

        return bv_ids