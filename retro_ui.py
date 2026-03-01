"""
RetroUI: ASCII-only formatting utilities for 1970s terminal compatibility.

Converts modern Unicode box-drawing characters and symbols to ASCII equivalents
while maintaining visual clarity. Useful for older terminals, teletypes, and
systems without Unicode support.

All output is guaranteed to use only ASCII characters (32-126, 9-13).
"""


class RetroUIFormatter:
    """Utilities for formatting output in ASCII-only mode."""

    @staticmethod
    def to_ascii_symbol(char):
        """
        Convert Unicode symbols to ASCII equivalents.

        Args:
            char (str): Single Unicode character or string

        Returns:
            str: ASCII-safe equivalent
        """
        symbol_map = {
            '✓': '[OK]',
            '✗': '[FAIL]',
            '⚠': '[WARN]',
            '→': '->',
            '↑': '[UP]',
            '↓': '[DN]',
            '•': '*',
            '◦': 'o',
            '◇': '<>',
            '■': '[#]',
            '□': '[ ]',
        }
        if char in symbol_map:
            return symbol_map[char]
        return char

    @staticmethod
    def section_header(text, width=70):
        """
        Create a section header with ASCII decoration.

        Args:
            text (str): Header text
            width (int): Total width of header (default 70)

        Returns:
            str: Formatted header with top and bottom bars
        """
        return f"{'=' * width}\n  {text}\n{'=' * width}"

    @staticmethod
    def simple_table(headers, rows, col_widths=None):
        """
        Create an ASCII table with simple borders.

        Args:
            headers (list): Column header strings
            rows (list of list): Table rows with values
            col_widths (list): Optional column widths (auto-calculated if None)

        Returns:
            str: Formatted ASCII table
        """
        if not col_widths:
            col_widths = [
                max(len(str(headers[i])), max(len(str(row[i])) for row in rows))
                for i in range(len(headers))
            ]

        output = []
        separator = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'

        # Top border
        output.append(separator)

        # Header row
        header_cells = [
            ' ' + str(h).ljust(col_widths[i]) + ' '
            for i, h in enumerate(headers)
        ]
        output.append('|' + '|'.join(header_cells) + '|')

        # Separator after header
        output.append(separator)

        # Data rows
        for row in rows:
            cells = [
                ' ' + str(row[i]).ljust(col_widths[i]) + ' '
                for i in range(len(row))
            ]
            output.append('|' + '|'.join(cells) + '|')

        # Bottom border
        output.append(separator)

        return '\n'.join(output)

    @staticmethod
    def box(title, content_lines, width=50):
        """
        Create a simple ASCII box around content.

        Args:
            title (str): Title for the box
            content_lines (list): Lines of text to include in box
            width (int): Box width (default 50)

        Returns:
            str: Formatted ASCII box
        """
        output = []
        output.append('+' + '-' * (width - 2) + '+')
        output.append('| ' + title.ljust(width - 4) + ' |')
        output.append('+' + '-' * (width - 2) + '+')

        for line in content_lines:
            # Truncate or pad lines to fit box width
            padded = str(line)[: width - 4].ljust(width - 4)
            output.append('| ' + padded + ' |')

        output.append('+' + '-' * (width - 2) + '+')
        return '\n'.join(output)

    @staticmethod
    def status_line(label, value, symbol=None, width=50):
        """
        Create a single status line with optional symbol.

        Args:
            label (str): Label text
            value (str): Value text
            symbol (str): Optional status symbol (✓, ✗, ⚠, etc.)
            width (int): Total line width (default 50)

        Returns:
            str: Formatted status line
        """
        if symbol:
            symbol = RetroUIFormatter.to_ascii_symbol(symbol)
        else:
            symbol = ''

        space = width - len(label) - len(str(value)) - len(symbol) - 3
        return f"{label}{' ' * max(1, space)}{str(value)}{symbol}"

    @staticmethod
    def sanitize(text):
        """
        Remove or replace all non-ASCII characters in text.

        Args:
            text (str): Input text

        Returns:
            str: Text with Unicode characters converted to ASCII equivalents
        """
        result = []
        for char in str(text):
            if ord(char) > 127:  # Non-ASCII
                result.append(RetroUIFormatter.to_ascii_symbol(char))
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def horizontal_line(char='-', width=70):
        """
        Create a horizontal line.

        Args:
            char (str): Character to use (default '-')
            width (int): Line width (default 70)

        Returns:
            str: Horizontal line
        """
        return char * width


# Convenience functions for common operations
def box_table(title, headers, rows):
    """Shorthand: create a bordered table."""
    return RetroUIFormatter.simple_table(headers, rows)


def ascii_safe(text):
    """Shorthand: sanitize text to ASCII-only."""
    return RetroUIFormatter.sanitize(text)


def header(text, width=70):
    """Shorthand: create a section header."""
    return RetroUIFormatter.section_header(text, width)
