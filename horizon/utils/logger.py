import colorama
import random
class Logger:
    def __init__(self, self_class):
        self.__color_class = colorama.Fore.MAGENTA

        colors = list(vars(colorama.Fore).values())
        self.__color_class = random.choice(colors)

        self.name_class = self_class.__class__.__name__

    def log(self, msg):
        print(self.__color_class + f'[{self.name_class}] ' + msg + colorama.Fore.RESET)
