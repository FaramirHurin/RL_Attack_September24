import heapq
from datetime import datetime


class PriorityQueue[T]:
    def __init__(self):
        self.elements = list[tuple[datetime, T]]()

    def is_empty(self):
        return len(self.elements) == 0

    def clear(self):
        self.elements.clear()

    def push(self, item: T, priority: datetime):
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        return heapq.heappop(self.elements)[1]

    def ppop(self):
        return heapq.heappop(self.elements)

    def size(self):
        return len(self.elements)

    def __len__(self):
        return len(self.elements)
