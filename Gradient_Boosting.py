import matplotlib.pyplot as plt
import sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def f_gb(x_nontest, y_nontest, x_test, y_test, tuning=True): #x är alla features utom stroke risk, y är enbart kolumnen för stroke risk.
  gb_model = GradientBoostingClassifier(
    n_estimators=200, #värde från boken
    max_depth=2, #värde från boken
    learning_rate=0.1, #värde från boken
    random_state=67) # värde: haha funny
  if tuning:
    x_train, x_val, y_train, y_val = train_test_split( x_nontest, y_nontest, test_size=0.2, random_state=67) #test_size = validation size (fördelning av 20% validering och 80% training)
    gb_model.fit(x_train, y_train)
    y_pred = gb_model.predict(x_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    return(gb_model, macro_f1)

  else:
    gb_model.fit(x_nontest, y_nontest)
    y_pred = gb_model.predict(x_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return(gb_model, macro_f1)
