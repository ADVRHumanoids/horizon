import colorama
import random
class Logger:
    def __init__(self, self_class, color=None):
        self.__color_class = colorama.Fore.MAGENTA

        blacklist = ['BLACK', 'LIGHTBLACK_EX', 'LIGHTWHITE_EX']
        colors = list()

        for name, color in vars(colorama.Fore).items():
            if name not in blacklist:
                colors.append(color)

        if not color:
            self.__color_class = random.choice(colors)
        else:
            self.__color_class = color

        self.name_class = self_class.__class__.__name__

    def log(self, msg):
        print(self.__color_class + f'[{self.name_class}] ' + msg + colorama.Fore.RESET)
