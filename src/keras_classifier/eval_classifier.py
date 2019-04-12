from src.keras_classifier.Calc import Calc

calc = Calc()
expected, predicted = calc.solve(123, '-', 321)
print('Expected=%s, Predicted=%s' % (expected, predicted))
