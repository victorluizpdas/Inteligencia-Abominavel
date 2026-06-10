#include <cmath>
#include <vector>
#include <iostream>
#include <format>

#include <opencv2/opencv.hpp>

#include "utils.h"


using namespace std;

double sigmoid(double x){
    return 1.0 /
     (1.0 + exp(-x));
}

double relu(double x) {
    return max(0.0, x);
}

double activation(vector<double>& weights, vector<double>& activations){
    double sum = 0;

    for(int i = 0; i < activations.size(); i++ ){
        sum += weights[i] * activations[i];
    }

    sum += weights[activations.size()];
    return sigmoid(sum);
}

double lossFunction(vector<double>& outputLayer, int label){
    double sum = 0;

    for(int i = 0; i < outputLayer.size(); i++){
        if( i == label){
            sum += pow(1.0 - outputLayer[i], 2);
        }else{
             sum += pow(0.0 - outputLayer[i], 2);
        }
    }

    return sum;
}

int output(vector<double>& outputLayer){
    double max = outputLayer[0];
    int resposta = 0;

    for(int i = 1; i < outputLayer.size(); i++){
        if(outputLayer[i] > max){
            max = outputLayer[i];
            resposta = i;
        }
    }

    return resposta;
}

int main() {
    
    auto images = readImages("dataset/train-images.idx3-ubyte");
    auto labels = readLabels("dataset/train-labels.idx1-ubyte");

    auto hiddenLayerWeights1 = initLayer(16, images[0].size() + 1);
    vector<double> hiddenLayer1(16);

    auto hiddenLayerWeights2 = initLayer(16, hiddenLayer1.size() + 1);
    vector<double> hiddenLayer2(16);

    auto outputLayerWeights =  initLayer(10, hiddenLayer2.size()+ 1);
    vector<double> outputLayer(10);

    for(int imageAt = 0; imageAt < images.size(); imageAt++){

        auto image = images[imageAt];
        auto label = labels[imageAt];

        for(int i = 0; i < hiddenLayer1.size(); i++){
            hiddenLayer1[i] = activation(hiddenLayerWeights1[i], image);
        }

        for(int i = 0; i < hiddenLayer2.size(); i++){
            hiddenLayer2[i] = activation(hiddenLayerWeights2[i], hiddenLayer1);
        }

        for(int i = 0; i < outputLayerWeights.size(); i++){
            outputLayer[i] = activation(outputLayerWeights[i], hiddenLayer2);
        }

        double loss = lossFunction(outputLayer,label);
        int resposta = output(outputLayer);
        if (imageAt % 1000 == 0 || imageAt == 0) {
            cout << fixed << setprecision(6);
            cout << "\n[Imagem n: " << imageAt << "] Perda: " << loss 
                 << " | Resposta Rede: " << resposta << " | Alvo Real: " << label;
        }
    }

    return 0;
}