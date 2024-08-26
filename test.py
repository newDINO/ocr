import zipfile

zip_ref = zipfile.ZipFile('data_gen/text_math.zip')

i = 9
file = zip_ref.open(f"data/texts/l1/{i}.png")
text = zip_ref.open('data/texts/l1/texts.txt').read().decode('utf-8').split('\n')[i]
print(text)

from PIL import Image
import matplotlib.pyplot as plt

image = Image.open(file)
plt.imshow(image)
plt.show()