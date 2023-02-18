#define FOR(i,n,m) for(short i=n; m>i; i++)
#define REVERSEFOR(i,n,m) for(short i=n; m<i; i--)
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>

using namespace std;

vector<double> randomWeights(vector<vector<double>> featuresMatrix);
vector<double> netProduct(const vector<vector<double>> &featuresMatrix, const vector<double> &weights);
vector<double> predict(const vector<vector<double>> &featuresMatrix, const vector<double> &weights);
vector<double> interpolate(const vector<vector<double>> &featuresMatrix, const vector<double> &weights);
pair<vector<double>, vector<double>> learn(vector<vector<double>> featuresMatrix, vector<double> y, double learningRate, int epochs);
double accuracy(vector<double> y, vector<double> predictedY);