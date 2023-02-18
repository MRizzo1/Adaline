#include "csvHandling.hpp"

/**
 * Lee un archivo csv y lo parsea en una matriz vectorial de doubles.
 *
 * @param path Direccion del archivo a leer.
 * @return Matriz vectorial de doubles representando la data.
 */
vector<vector<double>> readCsv(string path)
{
    vector<vector<double>> content;
    vector<double> row;
    string line, word;
    fstream file(path, ios::in);
    if (file.is_open())
    {
        while (getline(file, line))
        {
            row.clear();

            stringstream str(line);

            while (getline(str, word, ','))
                row.push_back(stod(word));
            content.push_back(row);
        }
    }
    else
        cout << "Could not open the file\n";

    return content;
}