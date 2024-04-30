(module
    (func
        (export "mystery")
    (param $n i32)
    (result i32)
    (local $a i32)
    (local $b i32)
    i32.const 0
    local.set $a
    i32.const 1
    local.set $b
    loop
        local.get $b
        local.get $b
        local.get $a
        i32.add
        local.set $b
        local.set $a
        local.get $n
        i32.const 1
        i32.sub
        local.set $n
        local.get $n
        br_if 0
    end
    local.get $a
    )
)