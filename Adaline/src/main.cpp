#include <iostream>
#include <stdio.h>
#include <string.h>
#include "adaline.hpp"
#include "csvHandling.hpp"

using namespace std;

int main(int argc, char *argv[])
{

   vector<vector<double>> earthSpace = readCsv("/home/mariangela/Downloads/EarthSpace.csv");
   vector<vector<double>> medSci = readCsv("/home/mariangela/Downloads/MedSci.csv");

   vector<double> y1 = std::vector<double>(earthSpace.size(), 1);
   vector<double> y2 = std::vector<double>(medSci.size(), -1);

   vector<vector<double>> X;
   vector<double> y;

   X.reserve(earthSpace.size() + medSci.size());
   X.insert(X.end(), earthSpace.begin(), earthSpace.end());
   X.insert(X.end(), medSci.begin(), medSci.end());

   y.reserve(y1.size() + y2.size());
   y.insert(y.end(), y1.begin(), y1.end());
   y.insert(y.end(), y2.begin(), y2.end());

   pair<vector<double>, vector<double>> result = learn(X, y, 1e-10, 100);
   vector<double> predictedY = predict(X, result.first);
   double acc = accuracy(y, predictedY);


   cout << "ACCURACY: " << acc << endl << endl;


   // THIRD QUESTION

   vector<vector<double>> datosT3 = readCsv("/home/mariangela/Downloads/datosT3.csv");

   vector<vector<double>> X2;
   vector<double> y3;

   for (vector<double> row : datosT3)
   {

      X2.push_back({row[0], log10(row[0] + 20000)});
      y3.push_back(row[1]);
   }

   result = learn(X2, y3, 1e-10, 20000);

   cout << "WEIGHTS [";
   for (double w : result.first)
   {
      cout << w << " ";
   }
   cout << "]" << endl
        << endl;

   vector<vector<double>> X3;
   vector<double> y4;

   for (vector<double> row : datosT3)
   {

      X3.push_back({row[0], pow(row[0], 3)});
      y4.push_back(row[1]);
   }

   result = learn(X3, y4, 1e-10, 10);

   cout << "WEIGHTS [";
   for (double w : result.first)
   {
      cout << w << " ";
   }
   cout << "]" << endl
        << endl;

   return 0;
}