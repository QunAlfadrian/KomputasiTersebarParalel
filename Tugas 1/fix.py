from PIL import Image

# Open the image using PIL
image = Image.open("larger.png")

# Remove color profile by saving without any extra information
image.save("larger_fixed.png", icc_profile=None)
