#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <functional>
#include <cctype>
using namespace std;

typedef vector<double> Vec;
typedef vector<vector<double>> Matrix;

// =====================================================================
//  Utilidades matematicas hechas a mano (sin librerias externas):
//  estadistica basica y procesamiento de matrices.
// =====================================================================
namespace MathUtils {

    // Media aritmetica de un vector.
    double mean(const Vec& v) {
        double s = 0.0;
        for (size_t i = 0; i < v.size(); i++) s += v[i];
        return s / v.size();
    }

    // Desviacion estandar poblacional de un vector dada su media m.
    double stdDev(const Vec& v, double m) {
        double s = 0.0;
        for (size_t i = 0; i < v.size(); i++) s += (v[i] - m) * (v[i] - m);
        return sqrt(s / v.size());
    }

    // Extrae la columna j de una matriz.
    Vec column(const Matrix& M, size_t j) {
        Vec c(M.size());
        for (size_t i = 0; i < M.size(); i++) c[i] = M[i][j];
        return c;
    }

    // Transpuesta de una matriz (n x d  ->  d x n).
    Matrix transpose(const Matrix& M) {
        Matrix T(M[0].size(), Vec(M.size()));
        for (size_t i = 0; i < M.size(); i++)
            for (size_t j = 0; j < M[0].size(); j++)
                T[j][i] = M[i][j];
        return T;
    }

    // Producto matriz-vector: (n x d) * (d) -> (n).
    Vec matVec(const Matrix& M, const Vec& v) {
        Vec r(M.size(), 0.0);
        for (size_t i = 0; i < M.size(); i++)
            for (size_t j = 0; j < v.size(); j++)
                r[i] += M[i][j] * v[j];
        return r;
    }
}

// Verifica que X no este vacia y que todas sus filas tengan el mismo numero de columnas.
void checkMatrix(const Matrix& X, const string& contexto) {
    if (X.empty())
        throw invalid_argument(contexto + ": la matriz X esta vacia");
    size_t d = X[0].size();
    if (d == 0)
        throw invalid_argument(contexto + ": X no tiene columnas");
    for (size_t i = 1; i < X.size(); i++)
        if (X[i].size() != d)
            throw invalid_argument(contexto + ": la fila " + to_string(i) + " tiene " +
                                   to_string(X[i].size()) + " columnas y se esperaban " + to_string(d));
}

// =====================================================================
//  Clase LinearRegression: regresion lineal multiple con gradiente descendente
// =====================================================================
class LinearRegression {
private:
    Vec weights;        // pesos del modelo (uno por variable de entrada)
    double bias;        // termino independiente (intercepto)
    double learningRate;// tasa de aprendizaje del gradiente descendente
    int epochs;         // numero de iteraciones de entrenamiento
    Vec means;          // media de cada columna (calculada con el train)
    Vec stds;           // desviacion de cada columna (calculada con el train)

public:
    // Constructor: lr = tasa de aprendizaje (>0), epochs = iteraciones (>0).
    LinearRegression(double lr = 0.01, int epochs = 1000)
        : bias(0.0), learningRate(lr), epochs(epochs) {
        if (lr <= 0) throw invalid_argument("constructor: la tasa de aprendizaje debe ser mayor que 0");
        if (epochs <= 0) throw invalid_argument("constructor: las epocas deben ser mayores que 0");
    }

    // Estandariza X modificandola: z = (x - media) / desviacion.
    // fitScaler = true: calcula y guarda media/desviacion (usar con datos de train).
    // fitScaler = false: reutiliza las guardadas (usar con datos de test).
    void data_scaling(Matrix& X, bool fitScaler = true) {
        checkMatrix(X, "data_scaling");
        size_t d = X[0].size();

        if (fitScaler) {
            means.assign(d, 0.0);
            stds.assign(d, 0.0);
            for (size_t j = 0; j < d; j++) {
                Vec col = MathUtils::column(X, j);
                means[j] = MathUtils::mean(col);
                stds[j] = MathUtils::stdDev(col, means[j]);
                if (stds[j] == 0) stds[j] = 1;   // evita dividir por cero
            }
        } else {
            if (means.empty())
                throw logic_error("data_scaling: primero debe escalar los datos de entrenamiento (fitScaler = true)");
            if (d != means.size())
                throw invalid_argument("data_scaling: X tiene " + to_string(d) +
                                       " columnas y el escalador se ajusto con " + to_string(means.size()));
        }

        for (size_t i = 0; i < X.size(); i++)
            for (size_t j = 0; j < d; j++)
                X[i][j] = (X[i][j] - means[j]) / stds[j];
    }

    // Entrena el modelo con gradiente descendente. X debe venir ya escalada.
    //   gradW = (2/n) * Xt * err ,  gradB = (2/n) * sum(err)
    void fit(const Matrix& X, const Vec& y) {
        checkMatrix(X, "fit");
        if (y.size() != X.size())
            throw invalid_argument("fit: X tiene " + to_string(X.size()) + " filas y y tiene " +
                                   to_string(y.size()) + " valores");

        size_t n = X.size();
        size_t d = X[0].size();
        weights.assign(d, 0.0);
        bias = 0.0;
        Matrix Xt = MathUtils::transpose(X);   // se calcula una sola vez

        for (int e = 0; e < epochs; e++) {
            Vec yhat = MathUtils::matVec(X, weights);
            Vec err(n);
            double gradB = 0.0;
            for (size_t i = 0; i < n; i++) {
                err[i] = (yhat[i] + bias) - y[i];
                gradB += err[i];
            }
            Vec gradW = MathUtils::matVec(Xt, err);

            for (size_t j = 0; j < d; j++)
                weights[j] -= learningRate * (2.0 / n) * gradW[j];
            bias -= learningRate * (2.0 / n) * gradB;
        }
    }

    // Predice la salida de cada fila de X (ya escalada): bias + sum(w_j * x_j).
    Vec predict(const Matrix& X) const {
        if (weights.empty())
            throw logic_error("predict: el modelo no ha sido entrenado, llame a fit() primero");
        checkMatrix(X, "predict");
        if (X[0].size() != weights.size())
            throw invalid_argument("predict: X tiene " + to_string(X[0].size()) +
                                   " columnas y el modelo se entreno con " + to_string(weights.size()));
        Vec yhat = MathUtils::matVec(X, weights);
        for (size_t i = 0; i < yhat.size(); i++) yhat[i] += bias;
        return yhat;
    }

    // Error de las predicciones: error cuadratico medio MSE = (1/n) * sum((yhat - y)^2).
    double score(const Matrix& X, const Vec& y) const {
        Vec yhat = predict(X);
        if (y.size() != yhat.size())
            throw invalid_argument("score: hay " + to_string(yhat.size()) + " predicciones y " +
                                   to_string(y.size()) + " valores reales");
        double sse = 0.0;
        for (size_t i = 0; i < y.size(); i++) sse += (yhat[i] - y[i]) * (yhat[i] - y[i]);
        return sse / y.size();
    }

    // Coeficiente de determinacion R^2 = 1 - SSE/SST (1 = ajuste perfecto).
    double r2_score(const Matrix& X, const Vec& y) const {
        Vec yhat = predict(X);
        if (y.size() != yhat.size())
            throw invalid_argument("r2_score: hay " + to_string(yhat.size()) + " predicciones y " +
                                   to_string(y.size()) + " valores reales");
        double meanY = MathUtils::mean(y);
        double sse = 0.0, sst = 0.0;
        for (size_t i = 0; i < y.size(); i++) {
            sse += (y[i] - yhat[i]) * (y[i] - yhat[i]);
            sst += (y[i] - meanY) * (y[i] - meanY);
        }
        if (sst == 0) return 0.0;   // evita dividir por cero
        return 1.0 - sse / sst;
    }

    // Consultas del estado interno del modelo.
    Vec getWeights() const { return weights; }
    double getBias() const { return bias; }
    Vec getMeans() const { return means; }
    Vec getStds() const { return stds; }
    // Direccion del buffer interno de weights (vive en el heap).
    const double* getWeightsAddress() const { return weights.data(); }
};

// =====================================================================
//  Lectura de datos y division train/test
// =====================================================================

// Convierte un texto a numero; falla si queda texto sobrante (por ejemplo "8abc").
double parseNumber(const string& cell, int numLinea) {
    size_t pos = 0;
    double v = 0.0;
    try {
        v = stod(cell, &pos);
    } catch (const exception&) {
        throw runtime_error("linea " + to_string(numLinea) + ": '" + cell + "' no es un numero");
    }
    while (pos < cell.size() && isspace(static_cast<unsigned char>(cell[pos]))) pos++;
    if (pos != cell.size())
        throw runtime_error("linea " + to_string(numLinea) + ": '" + cell + "' no es un numero valido");
    return v;
}

// Lee datos desde un flujo: salta el encabezado, la ultima columna es y.
// Detecta el separador (; , o tabulador) y acepta decimales con coma si el separador es ';'.
void loadCSV(istream& in, Matrix& X, Vec& y) {
    string line;
    int numLinea = 1;
    getline(in, line);                         // saltar encabezado
    size_t columnas = 0;

    while (getline(in, line)) {
        numLinea++;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        char sep = ',';
        if (line.find(';') != string::npos) sep = ';';
        else if (line.find('\t') != string::npos) sep = '\t';

        Vec row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, sep)) {
            if (sep == ';')
                for (size_t k = 0; k < cell.size(); k++) if (cell[k] == ',') cell[k] = '.';
            row.push_back(parseNumber(cell, numLinea));
        }

        if (row.size() < 2)
            throw runtime_error("linea " + to_string(numLinea) + ": se necesitan al menos 2 columnas (entrada y salida)");
        if (columnas == 0) columnas = row.size();
        else if (row.size() != columnas)
            throw runtime_error("linea " + to_string(numLinea) + ": tiene " + to_string(row.size()) +
                                " columnas y se esperaban " + to_string(columnas));

        y.push_back(row.back());               // ultima columna = salida
        row.pop_back();
        X.push_back(row);                      // resto = entradas
    }
    if (X.empty()) throw runtime_error("el archivo no contiene filas de datos");
}

// Lee datos desde un archivo.
void loadCSV(const string& filename, Matrix& X, Vec& y) {
    ifstream file(filename.c_str());
    if (!file.is_open()) throw runtime_error("no se pudo abrir el archivo " + filename);
    loadCSV(file, X, y);
}

// Division train/test: cada k-esima fila va a test, el resto a train.
void splitData(const Matrix& X, const Vec& y, int k,
               Matrix& Xtr, Vec& ytr, Matrix& Xte, Vec& yte) {
    if (k < 2) throw invalid_argument("splitData: k debe ser al menos 2");
    if (X.size() != y.size()) throw invalid_argument("splitData: X y y tienen distinto numero de filas");
    for (size_t i = 0; i < X.size(); i++) {
        if (i % k == 0) { Xte.push_back(X[i]); yte.push_back(y[i]); }
        else            { Xtr.push_back(X[i]); ytr.push_back(y[i]); }
    }
}

// =====================================================================
//  Pruebas
// =====================================================================

// ---------- Pruebas unitarias de los metodos ----------
void pruebaUnitaria() {
    cout << "=== Prueba data_scaling ===" << endl;
    LinearRegression scaler;
    Matrix A = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
    scaler.data_scaling(A, true);
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[i].size(); j++) cout << A[i][j] << " ";
        cout << endl;
    }

    cout << "\n=== Prueba fit (y = 2x + 1) ===" << endl;
    LinearRegression mlr(0.01, 1000);
    Matrix X = {{1}, {2}, {3}, {4}};
    Vec y = {3, 5, 7, 9};
    mlr.data_scaling(X, true);
    mlr.fit(X, y);
    cout << "weights: ";
    Vec w = mlr.getWeights();
    for (size_t i = 0; i < w.size(); i++) cout << w[i] << " ";
    cout << endl << "bias: " << mlr.getBias() << endl;

    cout << "\n=== Prueba predict y score ===" << endl;
    Matrix X_test = {{5}, {6}};
    mlr.data_scaling(X_test, false);
    Vec y_hat = mlr.predict(X_test);
    for (size_t i = 0; i < y_hat.size(); i++) cout << "y_hat: " << y_hat[i] << endl;
    Vec y1 = {11, 13};
    cout << "MSE (y_test exacto): " << mlr.score(X_test, y1) << endl;
    cout << "R2  (y_test exacto): " << mlr.r2_score(X_test, y1) << endl;
    Vec y2 = {10, 14};
    cout << "MSE (y_test con error): " << mlr.score(X_test, y2) << endl;
    cout << "R2  (y_test con error): " << mlr.r2_score(X_test, y2) << endl;
}

// ---------- Regresion simple con helados (+ extra x^2) ----------
void pruebaHelados() {
    cout << "=== Regresion simple: ice_cream.txt ===" << endl;

    Matrix X;
    Vec y;
    loadCSV("ice_cream.txt", X, y);
    cout << "Filas leidas: " << X.size() << endl;

    Matrix Xtr, Xte;
    Vec ytr, yte;
    splitData(X, y, 5, Xtr, ytr, Xte, yte);   // 20% test
    cout << "Train: " << Xtr.size() << "  Test: " << Xte.size() << endl;

    LinearRegression slr(0.1, 2000);
    slr.data_scaling(Xtr, true);
    slr.fit(Xtr, ytr);

    cout << "weights: ";
    Vec w = slr.getWeights();
    for (size_t i = 0; i < w.size(); i++) cout << w[i] << " ";
    cout << endl;
    cout << "bias: " << slr.getBias() << endl;

    slr.data_scaling(Xte, false);
    Vec y_hat = slr.predict(Xte);
    for (size_t i = 0; i < y_hat.size(); i++)
        cout << "y_hat: " << y_hat[i] << "   y_real: " << yte[i] << endl;

    cout << "MSE test: " << slr.score(Xte, yte) << endl;
    cout << "R2  test: " << slr.r2_score(Xte, yte) << endl;

    // Extra: agregar x^2 como segunda variable
    cout << "\n=== Extra: regresion con [x, x^2] ===" << endl;
    Matrix X2, Xtr2, Xte2;
    Vec ytr2, yte2;
    for (size_t i = 0; i < X.size(); i++) {
        Vec fila;
        fila.push_back(X[i][0]);
        fila.push_back(X[i][0] * X[i][0]);
        X2.push_back(fila);
    }
    splitData(X2, y, 5, Xtr2, ytr2, Xte2, yte2);

    LinearRegression poly(0.1, 2000);
    poly.data_scaling(Xtr2, true);
    poly.fit(Xtr2, ytr2);
    poly.data_scaling(Xte2, false);

    cout << "weights: ";
    Vec w2 = poly.getWeights();
    for (size_t i = 0; i < w2.size(); i++) cout << w2[i] << " ";
    cout << endl;
    cout << "bias: " << poly.getBias() << endl;
    cout << "MSE test: " << poly.score(Xte2, yte2) << endl;
    cout << "R2  test: " << poly.r2_score(Xte2, yte2) << endl;
}

// ---------- Regresion multiple con estudiantes ----------
void pruebaEstudiantes() {
    cout << "=== Regresion multiple: student_scores.txt ===" << endl;

    Matrix X;
    Vec y;
    loadCSV("student_scores.txt", X, y);
    cout << "Filas leidas: " << X.size() << "  Variables: " << X[0].size() << endl;

    Matrix Xtr, Xte;
    Vec ytr, yte;
    splitData(X, y, 5, Xtr, ytr, Xte, yte);   // 20% test
    cout << "Train: " << Xtr.size() << "  Test: " << Xte.size() << endl;

    LinearRegression mlr(0.1, 2000);
    mlr.data_scaling(Xtr, true);
    mlr.fit(Xtr, ytr);

    cout << "weights: ";
    Vec w = mlr.getWeights();
    for (size_t i = 0; i < w.size(); i++) cout << w[i] << " ";
    cout << endl;
    cout << "bias: " << mlr.getBias() << endl;

    mlr.data_scaling(Xte, false);
    Vec y_hat = mlr.predict(Xte);
    size_t mostrar = y_hat.size() < 10 ? y_hat.size() : 10;   // solo 10 predicciones
    for (size_t i = 0; i < mostrar; i++)
        cout << "y_hat: " << y_hat[i] << "   y_real: " << yte[i] << endl;

    cout << "MSE test: " << mlr.score(Xte, yte) << endl;
    cout << "R2  test: " << mlr.r2_score(Xte, yte) << endl;
}

// ---------- Analisis de memoria ----------
LinearRegression modeloGlobal(0.1, 100);          // memoria GLOBAL (segmento de datos)

void imprimirModelo(const string& nombre, const LinearRegression& m) {
    cout << nombre << endl;
    cout << "  direccion del objeto : " << static_cast<const void*>(&m) << endl;
    cout << "  sizeof(objeto)       : " << sizeof(m) << " bytes" << endl;
    cout << "  buffer de weights    : " << static_cast<const void*>(m.getWeightsAddress()) << endl;
}

void pruebaMemoria() {
    cout << "=== Analisis de memoria ===" << endl;

    // Datos de ejemplo para que weights tenga contenido (2 variables)
    Matrix X = {{1, 2}, {2, 1}, {3, 5}, {4, 3}};
    Vec y = {5, 4, 12, 10};

    // 1) GLOBAL
    Matrix Xg = X;
    modeloGlobal.data_scaling(Xg, true);
    modeloGlobal.fit(Xg, y);

    // 2) LOCAL (stack)
    LinearRegression modeloLocal(0.1, 100);
    Matrix Xl = X;
    modeloLocal.data_scaling(Xl, true);
    modeloLocal.fit(Xl, y);

    // 3) DINAMICA (heap)
    LinearRegression* modeloDinamico = new LinearRegression(0.1, 100);
    Matrix Xd = X;
    modeloDinamico->data_scaling(Xd, true);
    modeloDinamico->fit(Xd, y);

    imprimirModelo("[Global]", modeloGlobal);
    imprimirModelo("[Local - stack]", modeloLocal);
    imprimirModelo("[Dinamico - heap]", *modeloDinamico);
    cout << "  (la variable puntero 'modeloDinamico' vive en el stack: "
         << static_cast<const void*>(&modeloDinamico)
         << ", y apunta a " << static_cast<const void*>(modeloDinamico) << ")" << endl;

    // Tamanos de cada pieza
    cout << "\n--- Tamanos de los componentes ---" << endl;
    cout << "sizeof(double)                 : " << sizeof(double) << endl;
    cout << "sizeof(int)                    : " << sizeof(int) << endl;
    cout << "sizeof(vector<double>)         : " << sizeof(vector<double>) << endl;
    cout << "sizeof(LinearRegression)       : " << sizeof(LinearRegression) << endl;
    cout << "sizeof(LinearRegression*)      : " << sizeof(LinearRegression*) << endl;
    cout << "Suma sin padding: 3*" << sizeof(vector<double>) << " + 2*" << sizeof(double)
         << " + " << sizeof(int) << " = "
         << 3 * sizeof(vector<double>) + 2 * sizeof(double) + sizeof(int) << endl;

    // Vector (std::vector) de LinearRegression en memoria dinamica
    cout << "\n--- Vector de LinearRegression en el heap ---" << endl;
    int N = 5;
    vector<LinearRegression>* vec = new vector<LinearRegression>(N, LinearRegression(0.1, 100));

    cout << "N                                 : " << N << endl;
    cout << "sizeof(vector<LinearRegression>)  : " << sizeof(*vec) << " bytes (solo el 'encabezado')" << endl;
    cout << "Bytes de los elementos en el heap : " << vec->size() * sizeof(LinearRegression)
         << " (" << vec->size() << " x " << sizeof(LinearRegression) << ")" << endl;
    cout << "Direccion del objeto vector       : " << static_cast<const void*>(vec) << endl;
    cout << "Direccion del primer elemento     : " << static_cast<const void*>(&(*vec)[0]) << endl;
    for (int i = 0; i < N; i++)
        cout << "  elemento " << i << " en " << static_cast<const void*>(&(*vec)[i]) << endl;
    cout << "Distancia entre elementos         : "
         << reinterpret_cast<const char*>(&(*vec)[1]) - reinterpret_cast<const char*>(&(*vec)[0])
         << " bytes" << endl;

    // Arreglo dinamico clasico (new[]) de LinearRegression
    cout << "\n--- Arreglo dinamico new LinearRegression[3] ---" << endl;
    int M = 3;
    LinearRegression* arr = new LinearRegression[M];
    cout << "Bytes del bloque en el heap       : " << M * sizeof(LinearRegression)
         << " (" << M << " x " << sizeof(LinearRegression) << ")" << endl;
    cout << "sizeof(puntero arr)               : " << sizeof(arr) << " bytes (no conoce el tamano del bloque)" << endl;
    cout << "Direccion del bloque              : " << static_cast<const void*>(arr) << endl;
    for (int i = 0; i < M; i++)
        cout << "  arr[" << i << "] en " << static_cast<const void*>(&arr[i]) << endl;
    cout << "Distancia entre elementos         : "
         << reinterpret_cast<const char*>(&arr[1]) - reinterpret_cast<const char*>(&arr[0])
         << " bytes" << endl;

    // Liberar memoria dinamica
    delete[] arr;
    delete vec;
    delete modeloDinamico;
}

// ---------- Pruebas de validacion: el programa avisa en lugar de caerse ----------
void caso(const string& nombre, const function<void()>& f) {
    cout << nombre << ": ";
    try {
        f();
        cout << "no hubo error" << endl;
    } catch (const exception& e) {
        cout << "Error controlado -> " << e.what() << endl;
    }
}

void pruebaValidaciones() {
    cout << "=== Pruebas de validacion ===" << endl;

    caso("1. fit con X vacia", []() {
        LinearRegression m; Matrix X; Vec y; m.fit(X, y);
    });
    caso("2. escalar el test sin escalar antes el train", []() {
        LinearRegression m; Matrix X = {{1, 2}, {3, 4}}; m.data_scaling(X, false);
    });
    caso("3. predict sin entrenar", []() {
        LinearRegression m; Matrix X = {{1}}; m.predict(X);
    });
    caso("4. predict con distinto numero de columnas", []() {
        LinearRegression m(0.1, 10);
        Matrix X = {{1, 2}, {3, 4}, {5, 6}}; Vec y = {1, 2, 3};
        m.data_scaling(X, true); m.fit(X, y);
        Matrix Z = {{1}}; m.predict(Z);
    });
    caso("5. score con y de distinto tamano", []() {
        LinearRegression m(0.1, 10);
        Matrix X = {{1}, {2}, {3}}; Vec y = {1, 2, 3};
        m.data_scaling(X, true); m.fit(X, y);
        Vec y2 = {1, 2, 3, 4, 5}; m.score(X, y2);
    });
    caso("6. archivo con texto en lugar de numeros", []() {
        istringstream in("a,b\nx,y\n"); Matrix X; Vec y; loadCSV(in, X, y);
    });
    caso("7. archivo con filas de distinto largo", []() {
        istringstream in("a,b,c\n1,2,3\n4,5\n"); Matrix X; Vec y; loadCSV(in, X, y);
    });
    caso("8. tasa de aprendizaje invalida", []() {
        LinearRegression m(0.0, 100);
    });
    caso("9. archivo inexistente", []() {
        Matrix X; Vec y; loadCSV("no_existe.txt", X, y);
    });
}

// Ejecuta una prueba y captura cualquier error para que el programa no se caiga.
void ejecutar(void (*prueba)()) {
    try {
        prueba();
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
    }
}

int main() {
    ejecutar(pruebaUnitaria);
    cout << "\n------------------------------------------\n" << endl;
    ejecutar(pruebaHelados);
    cout << "\n------------------------------------------\n" << endl;
    ejecutar(pruebaEstudiantes);
    cout << "\n------------------------------------------\n" << endl;
    ejecutar(pruebaMemoria);
    cout << "\n------------------------------------------\n" << endl;
    ejecutar(pruebaValidaciones);
    return 0;
}