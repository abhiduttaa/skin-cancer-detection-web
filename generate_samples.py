import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

samples_dir = r"C:\Users\ABHIJIT DUTTA\.gemini\antigravity\scratch\cancer-detection-web\static\samples"
os.makedirs(samples_dir, exist_ok=True)

size = (300, 300)

# 1. Melanoma: Asymmetric, dark brownish-black, border irregularities
img1 = Image.new('RGB', size, color=(222, 195, 172))
draw1 = ImageDraw.Draw(img1)
# Base irregular lesion
draw1.ellipse([70, 75, 230, 220], fill=(45, 30, 32))
draw1.ellipse([110, 60, 215, 170], fill=(30, 22, 25))
draw1.ellipse([80, 130, 210, 235], fill=(65, 40, 42))
# Border protrusions / pigment network
for angle in np.linspace(0, 2*np.pi, 24):
    r = np.random.randint(65, 88)
    cx, cy = 150 + int(r * np.cos(angle)), 150 + int(r * np.sin(angle))
    rad = np.random.randint(8, 20)
    draw1.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=(40, 28, 30))

img1 = img1.filter(ImageFilter.GaussianBlur(radius=3))
img1.save(os.path.join(samples_dir, "melanoma_sample.jpg"), quality=95)

# 2. Benign Nevus (Common Mole): Symmetrical, smooth round border, uniform tan/light brown
img2 = Image.new('RGB', size, color=(230, 205, 185))
draw2 = ImageDraw.Draw(img2)
draw2.ellipse([90, 90, 210, 210], fill=(135, 85, 55))
draw2.ellipse([115, 115, 185, 185], fill=(105, 65, 40))
img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))
img2.save(os.path.join(samples_dir, "benign_nevus_sample.jpg"), quality=95)

# 3. Basal Cell Carcinoma: Pearly translucent pinkish-red nodule with arborizing vessels
img3 = Image.new('RGB', size, color=(225, 198, 180))
draw3 = ImageDraw.Draw(img3)
draw3.ellipse([85, 85, 215, 215], fill=(175, 115, 120))
draw3.ellipse([105, 105, 195, 195], fill=(195, 135, 140))
# Micro arborizing telangiectasia (reddish vessels)
for _ in range(8):
    x1, y1 = np.random.randint(110, 190), np.random.randint(110, 190)
    x2, y2 = x1 + np.random.randint(-30, 30), y1 + np.random.randint(-30, 30)
    draw3.line([(x1, y1), (x2, y2)], fill=(160, 45, 55), width=2)
img3 = img3.filter(ImageFilter.GaussianBlur(radius=3))
img3.save(os.path.join(samples_dir, "basal_cell_sample.jpg"), quality=95)

print("Generated 3 clinical sample dermatoscopy images in:", samples_dir)
