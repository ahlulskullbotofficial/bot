"""Identify simple national flags from colour layout. Moondream is too weak at this."""
from __future__ import annotations

from io import BytesIO

from PIL import Image

# Colours are approximate RGB; real photos and emoji flags will not match exactly.
VERTICAL = [
    ("France", [(0, 85, 164), (255, 255, 255), (239, 65, 53)]),
    ("Italy", [(0, 140, 69), (244, 245, 248), (205, 33, 42)]),
    ("Ireland", [(22, 155, 98), (255, 255, 255), (255, 121, 0)]),
    ("Belgium", [(0, 0, 0), (250, 224, 66), (237, 41, 57)]),
    ("Romania", [(0, 43, 127), (252, 209, 22), (206, 17, 38)]),
    ("Chad", [(0, 38, 100), (255, 204, 0), (210, 16, 52)]),
    ("Nigeria", [(0, 135, 81), (255, 255, 255), (0, 135, 81)]),
    ("Mali", [(20, 181, 58), (252, 209, 22), (206, 17, 38)]),
    ("Guinea", [(206, 17, 38), (252, 209, 22), (0, 148, 96)]),
    ("Senegal", [(0, 133, 63), (253, 239, 66), (227, 27, 35)]),
    ("Mexico", [(0, 104, 71), (255, 255, 255), (206, 17, 38)]),
    ("Canada", [(255, 0, 0), (255, 255, 255), (255, 0, 0)]),
    ("Peru", [(217, 16, 35), (255, 255, 255), (217, 16, 35)]),
]

HORIZONTAL = [
    ("Netherlands", [(174, 28, 40), (255, 255, 255), (33, 70, 139)]),
    ("Luxembourg", [(237, 41, 57), (255, 255, 255), (0, 161, 222)]),
    ("Russia", [(255, 255, 255), (0, 57, 166), (213, 43, 30)]),
    ("Germany", [(0, 0, 0), (221, 0, 0), (255, 206, 0)]),
    ("Armenia", [(217, 0, 18), (0, 51, 160), (242, 168, 0)]),
    ("Hungary", [(206, 41, 57), (255, 255, 255), (67, 111, 77)]),
    ("Bulgaria", [(255, 255, 255), (0, 150, 110), (214, 38, 18)]),
    ("Iran", [(35, 159, 73), (255, 255, 255), (218, 0, 0)]),
    ("Austria", [(237, 41, 57), (255, 255, 255), (237, 41, 57)]),
    ("Latvia", [(157, 34, 53), (255, 255, 255), (157, 34, 53)]),
    ("Ukraine", [(0, 87, 183), (255, 215, 0)]),
    ("Poland", [(255, 255, 255), (220, 20, 60)]),
    ("Indonesia", [(206, 17, 38), (255, 255, 255)]),
    ("Monaco", [(206, 17, 38), (255, 255, 255)]),
    ("Palestine", [(0, 0, 0), (255, 255, 255), (0, 122, 61)]),
    ("Pan-Arab", [(206, 17, 38), (255, 255, 255), (0, 0, 0)]),
    ("Estonia", [(0, 114, 206), (0, 0, 0), (255, 255, 255)]),
    ("Lithuania", [(253, 185, 19), (0, 106, 68), (193, 39, 45)]),
    ("Sierra Leone", [(30, 181, 58), (255, 255, 255), (0, 114, 198)]),
    ("Gabon", [(0, 158, 96), (252, 209, 22), (58, 117, 196)]),
    ("Ghana", [(206, 17, 38), (252, 209, 22), (0, 107, 61)]),
    ("Ethiopia", [(7, 137, 48), (252, 221, 9), (218, 18, 26)]),
    ("Colombia", [(255, 205, 0), (0, 48, 135), (206, 17, 38)]),
    ("Ecuador", [(255, 221, 0), (0, 82, 180), (237, 28, 36)]),
    ("Venezuela", [(255, 204, 0), (0, 61, 165), (207, 20, 43)]),
    ("Argentina", [(117, 170, 219), (255, 255, 255), (117, 170, 219)]),
    ("El Salvador", [(0, 71, 171), (255, 255, 255), (0, 71, 171)]),
    ("Nicaragua", [(0, 103, 198), (255, 255, 255), (0, 103, 198)]),
    ("Honduras", [(0, 188, 228), (255, 255, 255), (0, 188, 228)]),
    ("Thailand", [(237, 28, 36), (255, 255, 255), (36, 27, 141), (255, 255, 255), (237, 28, 36)]),
]

LOOKALIKES = {
    "Romania": ["Chad"],
    "Chad": ["Romania"],
    "Indonesia": ["Monaco"],
    "Monaco": ["Indonesia"],
    "Netherlands": ["Luxembourg"],
    "Luxembourg": ["Netherlands"],
    "Palestine": ["Jordan"],
    "Pan-Arab": ["Iraq", "Yemen", "Syria", "Egypt", "Sudan"],
    "Colombia": ["Ecuador"],
    "Ecuador": ["Colombia"],
}


def _open_rgb(image_bytes: bytes) -> Image.Image | None:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.seek(0)
        return image.convert("RGB")
    except Exception:
        return None


def _mean_color(image: Image.Image) -> tuple[float, float, float]:
    pixels = list(image.getdata())
    count = max(len(pixels), 1)
    r = sum(p[0] for p in pixels) / count
    g = sum(p[1] for p in pixels) / count
    b = sum(p[2] for p in pixels) / count
    return (r, g, b)


def _distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _bands(image: Image.Image, orientation: str, count: int) -> list[tuple[float, float, float]]:
    width, height = image.size
    colours = []
    for index in range(count):
        if orientation == "vertical":
            left = int(width * index / count)
            right = int(width * (index + 1) / count)
            crop = image.crop((left, 0, max(right, left + 1), height))
        else:
            top = int(height * index / count)
            bottom = int(height * (index + 1) / count)
            crop = image.crop((0, top, width, max(bottom, top + 1)))
        colours.append(_mean_color(crop))
    return colours


def _score(bands, target) -> float:
    if len(bands) != len(target):
        return 1e9
    return sum(_distance(band, colour) / 441.67 for band, colour in zip(bands, target)) / len(target)


def _hoist_is_red_triangle(image: Image.Image) -> bool:
    width, height = image.size
    hoist = image.crop((0, 0, max(int(width * 0.22), 1), height))
    rest = image.crop((int(width * 0.35), 0, width, height))
    hoist_colour = _mean_color(hoist)
    rest_colour = _mean_color(rest)
    hoist_red = hoist_colour[0] > hoist_colour[1] + 25 and hoist_colour[0] > hoist_colour[2] + 15
    rest_not_red = not (rest_colour[0] > rest_colour[1] + 25 and rest_colour[0] > rest_colour[2] + 15)
    return hoist_red and rest_not_red


def _looks_like_flag_photo(image: Image.Image) -> bool:
    """Skip ordinary photos/memes so colour averages do not invent a country."""
    width, height = image.size
    if height <= 0 or width < 24:
        return False
    ratio = width / height
    if ratio < 1.15 or ratio > 2.5:
        return False
    return True


def _bands_are_distinct(bands: list[tuple[float, float, float]]) -> bool:
    if len(bands) < 2:
        return False
    gaps = [_distance(bands[index], bands[index + 1]) for index in range(len(bands) - 1)]
    return min(gaps) >= 45


def _looks_like_japan(image: Image.Image) -> bool:
    width, height = image.size
    cx, cy = width // 2, height // 2
    radius = min(width, height) * 0.18
    disc = []
    field = []
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            pixel = image.getpixel((x, y))
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                disc.append(pixel)
            elif x < width * 0.12 or x > width * 0.88 or y < height * 0.12 or y > height * 0.88:
                field.append(pixel)
    if not disc or not field:
        return False
    disc_mean = tuple(sum(p[i] for p in disc) / len(disc) for i in range(3))
    field_mean = tuple(sum(p[i] for p in field) / len(field) for i in range(3))
    disc_red = disc_mean[0] > 140 and disc_mean[0] > disc_mean[1] + 40 and disc_mean[0] > disc_mean[2] + 40
    field_white = field_mean[0] > 200 and field_mean[1] > 200 and field_mean[2] > 200
    return disc_red and field_white


def _best_match(image: Image.Image, catalogue, orientation: str):
    ranked = []
    seen = set()
    for name, colours in catalogue:
        if name in seen:
            continue
        seen.add(name)
        bands = _bands(image, orientation, len(colours))
        ranked.append((_score(bands, colours), name))
    ranked.sort()
    return ranked


def identify_flag(image_bytes: bytes) -> str:
    image = _open_rgb(image_bytes)
    if image is None:
        return ""
    image.thumbnail((180, 120))
    if image.size[0] < 20 or image.size[1] < 12:
        return ""

    if not _looks_like_flag_photo(image):
        return ""

    if _looks_like_japan(image):
        return "Possible flag layout: Japan (white field with a red disc). Use only if the picture is actually a flag."

    vertical = _best_match(image, VERTICAL, "vertical")
    horizontal = _best_match(image, HORIZONTAL, "horizontal")
    best_vertical = vertical[0]
    best_horizontal = horizontal[0]
    best_score, best_name = min(best_vertical, best_horizontal, key=lambda item: item[0])
    orientation = "vertical" if best_score == best_vertical[0] else "horizontal"
    catalogue = VERTICAL if orientation == "vertical" else HORIZONTAL
    target = next(colours for name, colours in catalogue if name == best_name)
    bands = _bands(image, orientation, len(target))
    if not _bands_are_distinct(bands):
        return ""

    # Photos and memes are noisy; only trust a clearly better geometric match.
    runner_up = vertical[1][0] if orientation == "vertical" else horizontal[1][0]
    if best_score > 0.16:
        return ""
    if runner_up - best_score < 0.04:
        other = vertical[1][1] if orientation == "vertical" else horizontal[1][1]
        return (
            f"Possible flag layout, uncertain: closest are {best_name} and {other} "
            f"({orientation} bands). Do not name a country unless the picture is clearly a flag."
        )

    if best_name == "Palestine":
        if _hoist_is_red_triangle(image):
            return (
                "Possible flag layout: Palestine or Jordan (black-white-green bands with a red hoist triangle). "
                "Jordan has a white star in the triangle. Use only if the picture is actually a flag."
            )
        return ""
    if best_name == "Pan-Arab":
        if _hoist_is_red_triangle(image):
            return (
                "Possible flag layout: Sudan (red-white-black bands with a green hoist triangle). "
                "Use only if the picture is actually a flag."
            )
        return (
            "Possible flag layout: pan-Arab red-white-black tricolour (Iraq, Yemen, Syria, or Egypt). "
            "Do not pick one country unless script or stars/eagle are readable."
        )

    lookalikes = LOOKALIKES.get(best_name, [])
    extra = f" Lookalikes: {', '.join(lookalikes)}." if lookalikes else ""
    return (
        f"Possible flag layout: {best_name} ({orientation} colour bands). "
        f"Use only if the picture is actually a flag.{extra}"
    )
