class Kindergarden(object):

    def __init__(self, best_of=3):
        self.best_of = best_of

        self.bestchilds = [('', 0) for _ in range(self.best_of)]

        self.worst = 0
        self.best = 0

    def add(self, child, acc):
        if acc > self.worst:
            for idx, y in enumerate(self.bestchilds):
                if y[1] < acc:
                    temp = self.bestchilds[idx]
                    self.bestchilds[idx] = (str(child), acc)
                    self.add(temp[0], temp[1])
                    break