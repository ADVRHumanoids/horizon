import colorama
import random
import warnings

class Logger:
    def __init__(self, self_class, color=None):
        blacklist = ['BLACK', 'LIGHTBLACK_EX', 'LIGHTWHITE_EX']
        valid_colors = {
            name.upper(): val
            for name, val in vars(colorama.Fore).items()
            if name.upper() not in blacklist
        }
        colors = list(valid_colors.values())

        def warn_colored(message):
            print(
                colorama.Fore.YELLOW + "[Logger Warning] " + message + colorama.Fore.RESET
            )

        if isinstance(color, str):
            color_upper = color.upper()
            if color_upper in valid_colors:
                self.__color_class = valid_colors[color_upper]
            else:
                warn_colored(
                    f"Invalid color '{color}'. Falling back to random color.\n"
                    f"Valid options are: {', '.join(valid_colors.keys())}"
                )
                self.__color_class = random.choice(colors)
        elif color in colors or color is None:
            self.__color_class = color or random.choice(colors)
        else:
            warn_colored(
                "Invalid color format. Falling back to random color.\n"
                "Use a valid colorama.Fore color or color name string."
            )
            self.__color_class = random.choice(colors)

        self.name_class = self_class.__class__.__name__

    def log(self, msg):
        print(self.__color_class + f'[{self.name_class}] ' + msg + colorama.Fore.RESET)