from delta import Compiler, Phase

source = "123"

c = Compiler('program')
c.realize(source, Phase.SYNTACTIC_ANALYSIS)

print(c.parse_tree_str)


c = Compiler('program')
c.realize(source)

print(c.wat_code)
print(c.result)