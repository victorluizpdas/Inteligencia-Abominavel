#include <random>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>


using namespace std;

vector<vector<double>>  initLayer(int layerSize, int lastLayerSize){

    vector<vector<double>> vec;
   
    double xMin = -1, xMax = 1;

    srand(time(NULL));

    for(int i = 0; i < layerSize; i++){

        vector<double> temp; 
        
        for(int j = 0; j <= lastLayerSize; j++){

            double x =  xMin + (xMax - xMin) * ((double)rand() / RAND_MAX);
            temp.push_back(x);

        }

        vec.push_back(temp);
    }

    return vec;
}



uint32_t reverseInt(uint32_t value)
{
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8 ) |
           ((value & 0x0000FF00) << 8 ) |
           ((value & 0x000000FF) << 24);
}



vector<vector<double>> readImages(const string& fileName)
{
    ifstream file(fileName, ios::binary);

    if(!file)
    {
        cerr << "Erro ao abrir arquivo." << endl;
        return {};
    }

    uint32_t magicNumber;
    uint32_t numImages;
    uint32_t numRows;
    uint32_t numCols;

    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numImages), 4);
    file.read(reinterpret_cast<char*>(&numRows), 4);
    file.read(reinterpret_cast<char*>(&numCols), 4);

    magicNumber = reverseInt(magicNumber);
    numImages   = reverseInt(numImages);
    numRows     = reverseInt(numRows);
    numCols     = reverseInt(numCols);

    vector<vector<double>> images(numImages,vector<double>(numRows * numCols));

    for(uint32_t i = 0; i < numImages; i++)
    {
        for(uint32_t j = 0; j < numRows * numCols; j++)
        {
            unsigned char pixel;

            file.read(reinterpret_cast<char*>(&pixel), 1);

            images[i][j] = (float)pixel / 255.0;
        }
    }

    return images;
}

vector<int> readLabels(const string& fileName)
{
    ifstream file(fileName, ios::binary);

    if(!file)
    {
        cerr << "Erro ao abrir arquivo." << endl;
        return {};
    }

    uint32_t magicNumber;
    uint32_t numLabels;

    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numLabels), 4);

    magicNumber = reverseInt(magicNumber);
    numLabels   = reverseInt(numLabels);

    vector<int> labels(numLabels);

    for(uint32_t i = 0; i < numLabels; i++)
    {
        unsigned char label;

        file.read(reinterpret_cast<char*>(&label), 1);

        labels[i] = label;
    }

    return labels;
}