;; Simple examples using webassembly text format

(module

  (func
    (export "meaning_of_life")
    (result i32)

    i32.const 42
  )

  (func
    (export "multiply")
    (param $a i32)
    (param $b i32)
    (result i32)

    local.get $a
    local.get $b
    i32.mul
  )

  (func
    (export "f_to_c")
    (param $f f64)
    (result f64)

    local.get $f
    f64.const 32
    f64.sub
    f64.const 5
    f64.mul
    f64.const 9
    f64.div

  )
)
