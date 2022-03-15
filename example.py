import re

sentence = '[江/j  峡/j  大道/n]nt 北京/n [国际/n  货币/n  基金/n  组织/n]nt [韩国/ns  财政/n  经济院/n]nt '
pattern = re.compile(r'\[(.*?)](\w*)')
print(pattern.sub(r'\1', sentence))