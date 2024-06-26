'''
    @desciption: This module contains the SemanticVisitor class and Semantic mistake class, which is responsible for the semantic analysis of the AST.
    @author: @Lunaespindola
    @date: 2024/04/30
    A01751117
'''

# import the PTNodeVisitor class from the arpeggio module
from arpeggio import PTNodeVisitor

class SemanticMistake(Exception):

    def __init__(self, message):
        super().__init__(f'Semantic error: {message}')


class SemanticVisitor(PTNodeVisitor):

    RESERVED_WORDS = ['true', 'false', 'var',
                      'if', 'else', 'while', 'do']

    def __init__(self, parser, **kwargs):
        super().__init__(**kwargs)
        self.__parser = parser
        self.__symbol_table = []

    def position(self, node):
        return self.__parser.pos_to_linecol(node.position)

    @property
    def symbol_table(self):
        return self.__symbol_table

    def visit_decimal(self, node, children):
        value = int(node.value)
        if value >= 2 ** 31:
            raise SemanticMistake(
                'Out of range decimal integer literal at position '
                f'{self.position(node)} => { value }'
            )
            
    def visit_binary(self, node, children):
        value = int(node.value[2:], 2)
        if value >= 2 ** 31:
            raise SemanticMistake(
                'Out of range binary integer literal at position '
                f'{self.position(node)} => { value }'
            )
            
    def visit_octal(self, node, children):
        value = int(node.value[2:], 8)
        if value >= 2 ** 31:
            raise SemanticMistake(
                'Out of range octal integer literal at position '
                f'{self.position(node)} => { value }'
            )
            
    def visit_hexadecimal(self, node, children):
        value = int(node.value[2:], 16)
        if value >= 2 ** 31:
            raise SemanticMistake(
                'Out of range hexadecimal integer literal at position '
                f'{self.position(node)} => { value }'
            )

    def visit_decl_variable(self, node, children):
        name = node.value
        if name in SemanticVisitor.RESERVED_WORDS:
            raise SemanticMistake(
                'Reserved word not allowed as variable name at position '
                f'{self.position(node)} => {name}'
            )
        if name in self.__symbol_table:
            raise SemanticMistake(
                'Duplicate variable declaration at position '
                f'{self.position(node)} => {name}'
            )
        self.__symbol_table.append(name)

    def visit_lhs_variable(self, node, children):
        name = node.value
        if name not in self.__symbol_table:
            raise SemanticMistake(
                'Assignment to undeclared variable at position '
                f'{self.position(node)} => {name}'
            )

    def visit_rhs_variable(self, node, children):
        name = node.value
        if name not in self.__symbol_table:
            raise SemanticMistake(
                'Undeclared variable reference at position '
                f'{self.position(node)} => {name}'
            )