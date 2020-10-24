import datetime
import time
import random


random.seed(int(str(hash(datetime.date.today())) + str(hash(time.time()))))
the_board = ['Alexander', 'Arne', 'Fangzhou', 'Jonas', 'Lucia', 'Meaghan', 'Noam Chomsky', 'Carl Gustav Jung',
             'Karen Horney',
             'Andrey Nikolaevich Kolmogorov', 'Albert Thater Camus', 'Luke Skywalker', 'Vera Demberg Winterfors']
random.shuffle(the_board)
for participant in (['==================== WELCOME TO THE GLORIOUS MEGA MEETING ====================']
                    + the_board + ['==============================================================================']):
    print("\033[98m {}\033[00m" .format(participant))
