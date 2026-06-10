#include <cmath>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils.h"


using namespace std;

double normalizeNeuron(double x){
    return 1 /
     (1 + exp(-x));
}

double activation(vector<double> weights, vector<double> activations){
    double sum = 0;

    for(int i = 0; i < weights.size() - 1; i++ ){
        sum += weights.at(i) * activations.at(i);
    }

    sum += weights.at(weights.size() - 1);
    return normalizeNeuron(sum);
}

double output(vector<double> outputLayer, int label){
    double sum = 0;

    for(int i = 0; i < outputLayer.size(); i++){
        if( i == label){
            sum += pow(1 - outputLayer[i], 2);
        }else{
             sum += pow(0 - outputLayer[i], 2);
        }
    }

    return sum;
}

int main() {
    
    auto images = readImages("dataset/train-images.idx3-ubyte");
    auto labels = readLabels("dataset/train-labels.idx1-ubyte");

    auto hiddenLayerWeights1 = initLayer(16, images[0].size());
    vector<double> hiddenLayer1;

    auto hiddenLayer2 = initLayer(16, hiddenLayer1.size());
    vector<double> hiddenLayer2;

    auto outputLayerWeights =  initLayer(10, hiddenLayer2.size());
    vector<double> outputLayer;

    for(auto image : images ){

    }

    return 0;
}