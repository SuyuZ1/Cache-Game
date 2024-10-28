from bilibili_spider import BilibiliSpider

if __name__ == '__main__':
    spider = BilibiliSpider('config/spider.yaml')
    spider.run(debug=False)