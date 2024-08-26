from PIL import Image, ImageDraw, ImageFont

# Create a blank image
image = Image.new('RGB', (256, 128), color = (255, 255, 255))

# Initialize ImageDraw
draw = ImageDraw.Draw(image)

# Define the text and font
text = "W" * 7
font = ImageFont.load_default()  # You can load a custom font using ImageFont.truetype

# Position to start the text
position = (10, 10)

# Add text to image
draw.text(position, text, fill=(0, 0, 0), font_size=30)

# Save or display the image
image.show()  # Opens the image in the default viewer
image.save("output_image.png")
