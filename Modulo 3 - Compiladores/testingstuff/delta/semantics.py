from arpeggio import PTNodeVisitor


class SemanticMistake(Exception):

    def __init__(self, message):
        super().__init__(f'Semantic error: {message}')


class SemanticVisitor(PTNodeVisitor):

    def __init__(self, parser, **kwargs):
        super().__init__(**kwargs)
        self.__parser = parser
        self.__symbol_table = []

    def position(self, node):
        return self.__parser.pos_to_linecol(node.position)

    @property
    def symbol_table(self):
        return self.__symbol_table
    
    # def visit_decimal(self, node, children):
    #     value = int(node.value)
    #     if value >= 2 ** 31:
    #         raise SemanticMistake(
    #             'Out of range decimal integer literal at position '
    #             f'{self.position(node)} => { value }'
    #         )
            
    # def visit_binary(self, node, children):
    #     value = int(node.value[2:], 2)  # Remove '#b' prefix and convert to decimal
    #     if value >= 2 ** 31:
    #         raise SemanticMistake(
    #             f'Out of range binary integer literal at position {self.position(node)} => {value}'
    #         )

    # def visit_octal(self, node, children):
    #     value = int(node.value[2:], 8)  # Remove '#o' prefix and convert to decimal
    #     if value >= 2 ** 31:
    #         raise SemanticMistake(
    #             f'Out of range octal integer literal at position {self.position(node)} => {value}'
    #         )

    # def visit_hexadecimal(self, node, children):
    #     value = int(node.value[2:], 16)  # Remove '#x' prefix and convert to decimal
    #     if value >= 2 ** 31:
    #         raise SemanticMistake(
    #             f'Out of range hexadecimal integer literal at position {self.position(node)} => {value}'
    #         )
