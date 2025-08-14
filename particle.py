import random
import colorsys

class SandParticle:
	"""Sand particle with color and simple falling behavior."""
	def __init__(self, color=None, base_color=None):
		"""Initialize particle color and base color."""
		if color is not None:
			self.color = color
		else:
			self.color = random_color((0.1, 0.12), (0.5, 0.7), (0.7, 0.9))
		if base_color is not None:
			self.base_color = base_color
		else:
			self.base_color = self.color

	def update(self, grid, row, column):
		"""Attempt to move down, then down-left or down-right, else stay."""
		if grid.is_cell_empty(row + 1, column):
			return row + 1, column
		else:
			offsets = [-1, 1]
			random.shuffle(offsets)
			for offset in offsets:
				new_column = column + offset
				if grid.is_cell_empty(row +1, new_column):
					return row + 1, new_column

		return row, column

def random_color(hue_range, saturation_range, value_range):
	"""Generate an RGB color from HSV ranges."""
	hue = random.uniform(*hue_range)
	saturation = random.uniform(*saturation_range)
	value = random.uniform(*value_range)
	r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
	return int(r * 255), int(g * 255), int(b * 255)
