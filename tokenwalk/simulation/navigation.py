"""Keyboard selection of real stage indices; independent of window and execution."""


class StageNavigator:
    def __init__(self, stages, index=0):
        blocks = sorted({stage.block for stage in stages})
        self.columns = tuple(
            tuple(i for i, stage in enumerate(stages) if stage.block == block)
            for block in blocks
        )
        index = min(max(index, 0), len(stages) - 1)
        self.column = next(
            c for c, indices in enumerate(self.columns) if index in indices
        )
        self.row = self.columns[self.column].index(index)

    @property
    def selected(self):
        return self.columns[self.column][self.row]

    def move(self, dx=0, dy=0):
        self.column = (self.column + dx) % len(self.columns)
        self.row = min(self.row, len(self.columns[self.column]) - 1)
        self.row = (self.row + dy) % len(self.columns[self.column])

    @property
    def visible_columns(self):
        start = max(0, min(self.column - 1, len(self.columns) - 3))
        return range(start, min(start + 3, len(self.columns)))
