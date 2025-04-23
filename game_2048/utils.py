# Updated utils.py with RGB rendering support
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image

# Color maps
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    'high': (60, 58, 50)
}

TEXT_COLOR = {
    2: (119, 110, 101),
    4: (119, 110, 101),
    'high': (249, 246, 242)
}


def print_board(board):
    size = board.shape[0]
    print("-" * (size * 6 + 1))
    for r in range(size):
        row_str = "|".join([f"{int(cell):^5}" if cell != 0 else "     " for cell in board[r]])
        print(f"|{row_str}|")
        print("-" * (size * 6 + 1))


def log_transform(board):
    transformed = np.copy(board).astype(float)
    mask = transformed > 0
    transformed[mask] = np.log2(transformed[mask])
    return transformed


def render_board_as_image(board, cell_size=100):
    size = board.shape[0]
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.axis('off')
    ax.set_aspect('equal')

    for r in range(size):
        for c in range(size):
            value = board[r, c]
            color = TILE_COLORS.get(value, TILE_COLORS['high'])
            rgb = tuple(c / 255 for c in color)
            rect = patches.Rectangle((c, size - r - 1), 1, 1, linewidth=1, edgecolor='gray', facecolor=rgb)
            ax.add_patch(rect)

            if value != 0:
                text_color = TEXT_COLOR.get(value, TEXT_COLOR['high'])
                tc_rgb = tuple(c / 255 for c in text_color)
                ax.text(c + 0.5, size - r - 0.5, str(value), fontsize=16, ha='center', va='center', color=tc_rgb)

    # Convert figure to numpy RGB array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    rgb_array = np.asarray(img)[..., :3]
    plt.close(fig)
    return rgb_array
