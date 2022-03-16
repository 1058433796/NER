import re

sentence = ' "/w [江/j  峡/j  大道/n]nt 北京/n [国际/n  货币/n  基金/n  组织/n]nt [韩国/ns  财政/n  经济院/n]nt  （/w  附/v  图片/n  １/m  张/q  ）/w 《/w  '
pattern = re.compile(r'\[(.*?)](\w*)')
sentence = pattern.sub(r'\1', sentence)
print(sentence)
pairs = re.findall(r'(\S*?)/(\w*)', sentence)
print(list(zip(*pairs)))