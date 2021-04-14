from Algorithms.ESARSA import ESARSA


class ESARSA_VI(ESARSA):

    def get_interest(self, s, a):
        if s in [(4,3),(4,4)] and a == self.task.ACTION_UP:
            return 1
        else:
            return 0.5
        if s in [(2, 0)] and a == self.task.ACTION_UP:
            return 0.1
        else:
            return 1
        if (s[0] == 0 and a == self.task.ACTION_UP) or (s[0] == 5 and a == self.task.ACTION_DOWN) \
                or (s[1] == 0 and a == self.task.ACTION_LEFT) or (s[1] == 8 and a == self.task.ACTION_RIGHT):
            return 0.8
        else:
            return 1
