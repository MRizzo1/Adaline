#include "adaline.hpp"

/**
 * Dada una matriz vectorial de caracteristicas, genera un vector random de pesos.
 *
 * @param featuresMatrix Matriz vectorial de caracteristicas (o estimulos).
 * @return Vector con pesos randomizados.
 */
vector<double> randomWeights(vector<vector<double>> featuresMatrix)
{

    random_device rd{};
    mt19937 gen{rd()};
    normal_distribution<double> dis(0.0, 0.01);

    vector<double> weights(featuresMatrix[0].size() + 1);
    generate(weights.begin(), weights.end(), [&]()
             { return dis(gen); });

    return weights;
}

/**
 * Producto interno de multiplicar la matriz de estimulos por los pesos.
 *
 * @param featuresMatrix Matriz vectorial de caracteristicas (o estimulos).
 * @param weight Vector de pesos.
 * @return Vector que representa el producto interno de ambos vectores.
 */
vector<double> netProduct(const vector<vector<double>> &featuresMatrix, const vector<double> &weights)
{
    vector<double> result;
    for (auto &v : featuresMatrix)
        result.push_back(inner_product(v.begin(), v.end(), ++weights.begin(), 0.0) + weights[0]);
    return result;
}


/**
 * Predice si un feature pertenece a una categoria o a otra.
 *
 * @param featuresMatrix Matriz vectorial de caracteristicas (o estimulos).
 * @param weight Vector de pesos.
 * @return Vector de 1 y -1 indicando la categoria de los estimulos.
 */
vector<double> predict(const vector<vector<double>> &featuresMatrix, const vector<double> &weights)
{
    vector<double> nP = netProduct(featuresMatrix, weights);
    vector<double> result;

    FOR(i, 0, nP.size())
    {
        if (nP[i] < 0.0)
        {
            result.push_back(-1);
        }
        else
        {
            result.push_back(1);
        }
    }

    return result;
}

/**
 * Suma de dos vectores
 *
 * @param a Sumando.
 * @param b Sumando.
 * @return Resultado de la suma.
 */
vector<double> sumVectors(vector<double> a, vector<double> b)
{
    transform(a.begin(), a.end(), b.begin(), a.begin(), plus<double>());
    return a;
}

/**
 * Resta de dos vectores
 *
 * @param a Primer elemento.
 * @param b Segundo elemento.
 * @return Resultado de la resta.
 */
vector<double> substractVectors(vector<double> a, vector<double> b)
{
    transform(a.begin(), a.end(), b.begin(), a.begin(), minus<double>());
    return a;
}

/**
 * Multiplicaion de dos vectores
 *
 * @param a Primer elemento.
 * @param b Segundo elemento.
 * @return Resultado de la multiplicacion.
 */
vector<double> multiplyVectors(vector<vector<double>> a, vector<double> b)
{
    vector<double> result(a.size());
    if (a[0].size() == b.size())
    {
        FOR(i, 0, a.size())
        {

            FOR(k, 0, a[0].size())
            result[i] += a[i][k] * b[k];
        }
    }

    return result;
}

/**
 * Multiplicaion de un vector por un escalar.
 *
 * @param a Primer elemento.
 * @param b Escalar.
 * @return Resultado de la multiplicacion.
 */
vector<double> vectorByScalar(vector<double> a, double b)
{
    transform(a.begin(), a.end(), a.begin(), std::bind1st(std::multiplies<double>(), b));
    return a;
}

/**
 * Suma de todos los elementos que forman parte de un vector.
 *
 * @param a Vector.
 * @return Elementos del vector sumados.
 */
double vectorSum(vector<double> a)
{
    double result = 0.0;
    for (double i : a)
        result += i;

    return result;
}

/**
 * Elevar todos los elementos de un vector a un determinado exponente
 *
 * @param a Vector.
 * @param power Exponente.
 * @return Vector con la potencia de sus elementos al exponente indicado.
 */
vector<double> vectorPower(vector<double> a, int power)
{
    vector<double> result;
    for (double i : a)
        result.push_back(pow(i, power));

    return result;
}

/**
 * Obtiene la transpuesta de una matriz.
 *
 * @param a Matriz.
 * @return Transpuesta de la matriz.
 */
vector<vector<double>> transpose(vector<vector<double>> a)
{
    if (a.size() == 0)
        return {};

    vector<vector<double>> result(a[0].size(), vector<double>());

    FOR(i, 0, a.size())
    {
        FOR(j, 0, a[i].size())
        {
            result[j].push_back(a[i][j]);
        }
    }

    return result;
}

/**
 * Ajusta los pesos obtenidos a lo random inicialmente haciendo uso del gradiente y del minimo error.
 *
 * @param featuresMatrix Matriz vectorial de caracteristicas (o estimulos).
 * @param y Vector con las predicciones deseadas para cada fila de estimulo.
 * @param learningRate Tasa de aprendizaje.
 * @param epochs Numero de iteraciones a realizar.

 * @return Par con los pesos ajustados y los errores en cada iteracion.
 */
pair<vector<double>, vector<double>> learn(vector<vector<double>> featuresMatrix, vector<double> y, double learningRate, int epochs)
{
    vector<double> mse_it;
    vector<double> weights = randomWeights(featuresMatrix);

    FOR(i, 0, epochs)
    {
        vector<double> nP = netProduct(featuresMatrix, weights);
        vector<double> gradient = vectorByScalar(substractVectors(nP, y), 2);

        vector<double> w(++weights.begin(), weights.end());

        vector<double> wSum = vectorByScalar(multiplyVectors(transpose(featuresMatrix), vectorByScalar(gradient, -1.0)), learningRate);
        wSum = sumVectors(w, wSum);
        double w0 = weights[0] + learningRate * (-1) * vectorSum(gradient);



        vector<double> newWeights = {w0};
        newWeights.insert(newWeights.end(), wSum.begin(), wSum.end());
        weights = newWeights;

        double mse = vectorSum(vectorPower(substractVectors(y, nP), 2)) / y.size();
        mse_it.push_back(mse);

    }

    return {weights, mse_it};
}

/**
 * Regresa el porcentaje de acierto de una prediccion comparada con el valor deseado.
 *
 * @param y Vector con las predicciones deseadas para cada fila de estimulo.
 * @param predictedY Vector con las predicciones realizadas para cada fila de estimulos.

 * @return Porcentaje de acierto.
 */
double accuracy(vector<double> y, vector<double> predictedY){
    double correctPredictions = 0.0;

    FOR(i, 0, y.size()){
        if(y[i] == predictedY[i]){
            correctPredictions += 1.0;
        } 
    }
    
    double accuracy = (correctPredictions / y.size()) * 100;
    return accuracy;
}
