import os
from PIL import Image

def make_pos_txt(folder, output_file='pos.txt', img_ext='.jpg'):
    with open(output_file, 'w') as f:
        for file in os.listdir(folder):
            if file.endswith('.txt'):
                img_file = file.replace('.txt', img_ext)
                img_path = os.path.join(folder, img_file)
                txt_path = os.path.join(folder, file)

                if not os.path.exists(img_path):
                    print(f"not find : {img_path}")
                    continue

                with open(txt_path, 'r') as ann:
                    content = ann.read().strip()

                try:
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                except:
                    print(f"load fail : {img_path}")
                    continue
 
                tokens = content.split()
                if len(tokens) < 5:
                    print(f"label format error : {txt_path}")
                    continue

                num_objs = int(tokens[0])
                bboxes = tokens[1:]
                valid = True
                for i in range(num_objs):
                    x, y, w, h = map(int, bboxes[i*4:i*4+4])
                    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                        print(f"out of box : {img_file} ({x},{y},{w},{h})")
                        valid = False
                        break

                if valid:
                    line = f"{img_path} {content}\n"
                    f.write(line)
    print(" pos.txt")

make_pos_txt("pos2") 
