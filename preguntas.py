"""
Optimización usando gradiente descendente - Regresión polinomial
-----------------------------------------------------------------------------------------

En este laboratio se estimarán los parámetros óptimos de un modelo de regresión 
polinomial de grado `n`.

"""


def pregunta_01():
    """
    Complete el código presentado a continuación.
    """
    # Importe pandas
    import pandas as pd

    # Importe PolynomialFeatures
    from sklearn.preprocessing import PolynomialFeatures

    # Cargue el dataset `data.csv`
    data = pd.read_csv("data.csv")

    # Cree un objeto de tipo `PolynomialFeatures` con grado `2`
    poly = PolynomialFeatures(2)

    # Transforme la columna `x` del dataset `data` usando el objeto `poly`
    x_poly = poly.fit_transform(data[["x"]])

    # Retorne x y y
    return x_poly, data.y


def pregunta_02():

    # Importe numpy
    import numpy as np
    
    def gradiente(x, error):
  
      gradient_w2 = -2 * sum(
          [error * x_value[0] for error, x_value in zip(error, x)]
      )
      gradient_w1 = -2 * sum(
          [error * x_value[1] for error, x_value in zip(error, x)]
      )
      gradient_w0 = -2 * sum(
          [error * x_value[2] for error, x_value in zip(error, x)]
      )
      array = np.array( [gradient_w2,gradient_w1, gradient_w0])
      return array

    x_poly, y = pregunta_01()

    # Fije la tasa de aprendizaje en 0.0001 y el número de iteraciones en 1000
    learning_rate = 0.0001
    n_iterations = 500

    # Defina el parámetro inicial `params` como un arreglo de tamaño 3 con ceros
    params = np.zeros(x_poly.shape[1])
    for _ in range(n_iterations):

        # Compute el pronóstico con los parámetros actuales
        y_pred = np.dot(x_poly, params)

        # Calcule el error
        error =  y -y_pred

        # Calcule el gradiente
        gradient =gradiente(x_poly, error)#2 * np.dot(error, x_poly) / len(y)

        # Actualice los parámetros
        params = params - learning_rate * gradient

    return params
