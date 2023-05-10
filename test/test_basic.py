import sys
sys.path.append('../src')

from value import Value
import pytest


def approx(x):
	return pytest.approx(x)

def test_values_1():
	v1 = Value(2.3, label='v1')
	v2 = Value (3, label='v2')
	v3 = Value (-4.3, label='v3')

	assert(v1.data == 2.3)
	assert(v3.data == -4.3)
	print(f"Values are : {v1}, {v2}, {v3}")

def test_simple_ops():
	v1 = Value(2.3, label='v1')
	v2 = Value (3, label='v2')
	v3 = Value (-4.3, label='v3')

	assert((v1 + v2).data == 5.3)
	assert((3 + v2).data == 6)
	assert((v3 + 1).data == -3.3)

	assert((v1 - v2).data == approx(-0.7))
	assert((3 - v2).data == 0)
	assert((v2 - 0.5).data == approx(2.5))
	

	assert((v1 * v3).data == approx(-9.89))
	assert((0 * v3).data == 0)
	assert((v1 * 2).data == 4.6)

	assert((v2 / 1.5).data == 2)
	assert((8.6 / v3).data == -2)



test_values_1()
test_simple_ops()



