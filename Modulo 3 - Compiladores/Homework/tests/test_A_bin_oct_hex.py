from unittest import TestCase
from delta import Compiler, SyntaxMistake
from delta.semantics import SemanticMistake


class TestBinHexOctLiteral(TestCase):

    def setUp(self):
        self.c = Compiler('program')

    def test_syntax_mistake_1(self):
        with self.assertRaises(SyntaxMistake):
            self.c.realize('#b11002')

    def test_syntax_mistake_2(self):
        with self.assertRaises(SyntaxMistake):
            self.c.realize('#o431870')

    def test_syntax_mistake_3(self):
        with self.assertRaises(SyntaxMistake):
            self.c.realize('#x0ffice')

    def test_syntax_mistake_4(self):
        with self.assertRaises(SyntaxMistake):
            self.c.realize('#b')

    def test_syntax_mistake_5(self):
        with self.assertRaises(SyntaxMistake):
            self.c.realize('#o')

    def test_syntax_mistake_6(self):
        with self.assertRaises(SyntaxMistake):
            self.c.realize('#x')

    def test_0_bin(self):
        self.assertEqual(
            0,
            self.c.realize('#b0'))

    def test_0_oct(self):
        self.assertEqual(
            0,
            self.c.realize('#o0'))

    def test_0_hex(self):
        self.assertEqual(
            0,
            self.c.realize('#x0'))

    def test_bin(self):
        self.assertEqual(
            4784,
            self.c.realize('#b1001010110000'))

    def test_oct(self):
        self.assertEqual(
            342391,
            self.c.realize('#o1234567'))

    def test_hex(self):
        self.assertEqual(
            2048045518,
            self.c.realize('#x7A12b1CE'))

    def test_max_bin(self):
        self.assertEqual(
            2147483647,
            self.c.realize('#b1111111111111111111111111111111'))

    def test_max_oct(self):
        self.assertEqual(
            2147483647,
            self.c.realize('#o17777777777'))

    def test_max_hex(self):
        self.assertEqual(
            2147483647,
            self.c.realize('#x7FFFFFFF'))

    def test_bin_overflow(self):
        with self.assertRaises(SemanticMistake):
            self.c.realize('#b10000000000000000000000000000000')

    def test_oct_overflow(self):
        with self.assertRaises(SemanticMistake):
            self.c.realize('#o20000000000')

    def test_hex_overflow(self):
        with self.assertRaises(SemanticMistake):
            self.c.realize('#x80000000')