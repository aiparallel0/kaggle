class RetroUIFormatter:
    @staticmethod
    def bold(text):
        """Formats text to be bold in ASCII terminal."""
        return f'\033[1m{text}\033[0m'

    @staticmethod
    def underline(text):
        """Formats text to be underlined in ASCII terminal."""
        return f'\033[4m{text}\033[0m'

    @staticmethod
    def red(text):
        """Formats text to be red in ASCII terminal."""
        return f'\033[91m{text}\033[0m'

    @staticmethod
    def green(text):
        """Formats text to be green in ASCII terminal."""
        return f'\033[92m{text}\033[0m'

    @staticmethod
    def yellow(text):
        """Formats text to be yellow in ASCII terminal."""
        return f'\033[93m{text}\033[0m'

    # Additional formatting methods can be added here as needed
