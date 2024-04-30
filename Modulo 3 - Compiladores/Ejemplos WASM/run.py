from wasmtime import Store, Module, Instance


def call_wasm_fun(file_name, fn_name, *args):
    with open(file_name) as file:
        wat_code = file.read()
    store = Store()
    module = Module(store.engine, wat_code)
    instance = Instance(store, module, [])
    entry_function = instance.exports(store)[fn_name]
    return entry_function(store, *args)


def main():
    print('CALL OF 1 :',call_wasm_fun('test.wat', 'mystery', 1))
    print('CALL OF 2 :',call_wasm_fun('test.wat', 'mystery', 2))
    print('CALL OF 5 :',call_wasm_fun('test.wat', 'mystery', 5))
    print('CALL OF 7 :',call_wasm_fun('test.wat', 'mystery', 7))


if __name__ == '__main__':
    main()
