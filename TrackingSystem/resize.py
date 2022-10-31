from PIL import Image

image = Image.open('LEVANTAMENTO UM-Model-1.png')
new_image = image.resize((33090,23390))
new_image.save('newImage.png')

print(image.size) # Output: (1920, 1280)
print(new_image.size) # Output: (400, 400)