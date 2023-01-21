import os
from PIL.ImageFont import FreeTypeFont

FONT_URL = "https://github.com/fcakyon/ultralyticsplus/releases/download/0.0.16/OpenSans-Bold.ttf"


def get_font(size: int) -> FreeTypeFont:
    from PIL import ImageFont

    try:
        font = ImageFont.truetype("OpenSans-Bold.ttf", size)
    except OSError:
        from sahi.utils.file import download_from_url

        package_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(package_dir, "OpenSans-Bold.ttf")
        download_from_url(from_url=FONT_URL, to_path=font_path)
        font = ImageFont.truetype(font_path, size)
    return font


def add_text_to_image(
    text,
    pil_image,
    brightness: float = 0.75,
    text_font_size: int = 50,
    crop_margin: int = None,
):
    from PIL import ImageDraw, ImageEnhance

    RESIZE_WIDTH_TO = 1280
    MAX_TEXT_WIDTH = 860

    # Create an ImageEnhance object
    enhancer = ImageEnhance.Brightness(pil_image)

    # Darken the image by 25%
    pil_image = enhancer.enhance(brightness)

    # Resize the image
    pil_image = pil_image.resize((RESIZE_WIDTH_TO, RESIZE_WIDTH_TO))

    # crop image
    if crop_margin is not None:
        width, height = pil_image.size
        mid_height = int(height / 2)
        min_height = max(0, mid_height - crop_margin)
        max_height = min(height, mid_height + crop_margin)
        pil_image = pil_image.crop((0, min_height, width, max_height))

    # Create an ImageDraw object
    draw = ImageDraw.Draw(pil_image)

    # Define the font and text to be written
    font = get_font(size=text_font_size)

    # Get the bounding box of the text
    bbox = draw.textbbox((0, 0), text, font=font)

    # Get the width and height of the text
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # update font size
    if text_width > MAX_TEXT_WIDTH:
        text_font_size = int(MAX_TEXT_WIDTH / text_width * text_font_size)
        font = get_font(size=text_font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    # Define the coordinates of the smaller rounded rectangle
    box_margin = 20
    x1, y1 = (pil_image.width - text_width) / 2 - box_margin, (
        pil_image.height - text_height
    ) / 2 - box_margin / 3
    x2, y2 = (pil_image.width + text_width) / 2 + box_margin, (
        pil_image.height + text_height * 2
    ) / 2 + box_margin / 3

    # Define the radius of the rounded corners
    radius = 15

    # Draw the rounded rectangle
    draw.rounded_rectangle(
        [(x1, y1), (x2, y2)],
        radius=radius,
        fill=(255, 255, 255),
        outline=None,
        width=5,
    )

    # Draw the text on the image
    draw.text(
        ((pil_image.width - text_width) / 2, (pil_image.height - text_height) / 2),
        text,
        font=font,
        fill=(0, 0, 0),
    )

    return pil_image
