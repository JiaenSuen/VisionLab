import os

with open('neg.txt', 'w') as f:
    for file in os.listdir('neg'):
        if file.endswith('.jpg') or file.endswith('.png'):
            f.write(f'neg/{file}\n')
