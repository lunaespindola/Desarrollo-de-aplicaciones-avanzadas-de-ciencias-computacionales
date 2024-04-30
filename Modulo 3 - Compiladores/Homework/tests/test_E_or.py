# File: tests/test_E_or.py

from unittest import TestCase
from delta import Compiler, SyntaxMistake
from wasmtime._trap import Trap


class TestOr(TestCase):

    def setUp(self):
        self.c = Compiler('program')

    def test_syntax_mistake(self):
        with self.assertRaises(SyntaxMistake):
            self.c.realize('1 ||')

    def test_or1(self):
        self.assertEqual(1,
                         self.c.realize('true || true'))

    def test_or2(self):
        self.assertEqual(1,
                         self.c.realize('true || false'))

    def test_or3(self):
        self.assertEqual(1,
                         self.c.realize('false || true'))

    def test_or4(self):
        self.assertEqual(0,
                         self.c.realize('false || false'))

    def test_or5(self):
        self.assertEqual(1,
                         self.c.realize('5 || 0 || 3 || 2'))

    def test_or6(self):
        self.assertEqual(0,
                         self.c.realize('0 || 0 || 0 || 0'))

    def test_or7(self):
        self.assertEqual(0,
                         self.c.realize('!1 || !2 || !3'))

    def test_or8(self):
        self.assertEqual(1,
                         self.c.realize('!!1 || !!2 || !!3'))

    def test_or9(self):
        self.assertEqual(1,
                         self.c.realize('3 * 4 || 8 - 4 * 2'))

    def test_or10(self):
        self.assertEqual(0,
                         self.c.realize('!(3 * 4) || 8 - 4 * 2'))

    def test_or11(self):
        self.assertEqual(1,
                         self.c.realize('''
                         var t, u;
                         t = false;
                         u = true;
                         !u || !t || u/t
                         '''))

    def test_or_runtime_error(self):
        with self.assertRaises(Trap):
            self.c.realize('''
            var t, u;
            t = false;
            u = true;
            !u || t || u/t
            ''')