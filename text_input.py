import numpy as np

model_name = 'led_zeppelin.'

#text import
with open(str(model_name + 'txt'),'r',encoding='utf8') as f:
    text = f.read()

all_characters = set(text) #all unique characters
#decoder = dict(enumerate(all_characters)) #decoder
decoder = {0: 'M', 1: 'A', 2: 'N', 3: 'F', 4: 'e', 5: 'q', 6: 'l', 7: 'H', 8: 'E', 9: 'Q', 10: 'B', 11: ',', 12: '/', 13: '0', 14: 'I', 15: 'V', 16: 't', 17: 'x', 18: 'g', 19: 'J', 20: ']', 21: 'b', 22: 'd', 23: 'o', 24: 'z', 25: 'w', 26: 'S', 27: 'u', 28: 'L', 29: 'C', 30: '?', 31: '!', 32: 'y', 33: 'G', 34: 'f', 35: '7', 36: '\n', 37: 'W', 38: 'K', 39: 'r', 40: "'", 41: 'c', 42: ' ', 43: 'R', 44: '3', 45: 'v', 46: ':', 47: ')', 48: 'm', 49: 'Y', 50: '(', 51: 'a', 52: '-', 53: 'T', 54: '4', 55: '.', 56: '"', 57: 'k', 58: 'U', 59: 'j', 60: 'p', 61: 'D', 62: 'P', 63: '[', 64: '–', 65: 'i', 66: 'O', 67: '2', 68: 's', 69: 'h', 70: 'n'}
encoder = {char: ind for ind,char in decoder.items()} #encoder

encoded_text = np.array([encoder[char] for char in text])

#print(decoder)

