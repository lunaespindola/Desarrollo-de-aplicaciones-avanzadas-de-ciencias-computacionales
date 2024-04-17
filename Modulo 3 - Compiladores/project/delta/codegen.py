from arpeggio import PTNodeVisitor


class CodeGenerationVisitor(PTNodeVisitor):

    WAT_TEMPLATE = ''';; Code generated by the Delta compiler
(module
  (func
    (export "_start")
    (result i32)
{}  )
)
'''

    def __init__(self, symbol_table, **kwargs):
        super().__init__(**kwargs)
        self.__symbol_table = symbol_table
        
    def visit_program(self, node, children):
        return self.WAT_TEMPLATE.format(children[0])
    
    def visit_expression(self, node, children):
        return children[0]
