#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#include "utils.h"

using namespace std;

double sigmoid(double x){
    if(x >= 0){
        double z = exp(-x);
        return 1.0 / (1.0 + z);
    }

    double z = exp(x);
    return z / (1.0 + z);
}

double derSigmoidActivated(double y){
    return y * (1.0 - y);
}

double activation(vector<double>& weights,vector<double>& activations,double bias){
    double sum = bias;

    for(int i = 0; i < activations.size(); i++){
        sum += weights[i] * activations[i];
    }

    return sigmoid(sum);
}

double lossFunction(vector<double>& outputLayer, int label){
    double sum = 0.0;

    for(int i = 0; i < outputLayer.size(); i++)    {
        double target = (i == label) ? 1.0 : 0.0;

        double error = outputLayer[i] - target;

        sum += error * error;
    }

    return sum;
}

int output(vector<double>& outputLayer){
    int answer = 0;
    double maxValue = outputLayer[0];

    for(int i = 1; i < outputLayer.size(); i++){
        if(outputLayer[i] > maxValue){
            maxValue = outputLayer[i];
            answer = i;
        }
    }

    return answer;
}

vector<double> calcOutputDelta(vector<double>& output,int label){

    vector<double> delta(output.size());

    for(int i = 0; i < output.size(); i++){
        double target =
            (i == label) ? 1.0 : 0.0;

        delta[i] =
            2.0 *
            (output[i] - target) *
            derSigmoidActivated(output[i]);
    }

    return delta;
}

vector<double> calcHiddenDelta(vector<double>& activations,vector<double>& nextDelta,vector<vector<double>>& nextWeights){
    vector<double> delta(activations.size(),0.0);

    for(int i = 0; i < activations.size(); i++){
        double sum = 0.0;

        for(int j = 0; j < nextDelta.size(); j++){
            sum += nextWeights[j][i] * nextDelta[j];
        }

        delta[i] = sum * derSigmoidActivated(activations[i]);
    }

    return delta;
}

void updateWeights(vector<vector<double>>& weights,vector<double>& delta,vector<double>& previousActivations,double learningRate){
    for(int neuron = 0; neuron < weights.size();neuron++){

        for(int weight = 0;weight < weights[neuron].size();weight++){

            weights[neuron][weight] -=
                learningRate *
                delta[neuron] *
                previousActivations[weight];
        }
    }
}

void updateBiases(vector<double>& biases,vector<double>& delta,double learningRate){
    for(int i = 0; i < biases.size(); i++){

        biases[i] -= learningRate * delta[i];
    }
}

int main(){
    
    auto images =readImages("dataset/train-images.idx3-ubyte");
    auto labels =readLabels("dataset/train-labels.idx1-ubyte");

    auto hiddenLayerWeights1 =initLayer(16,images[0].size());
    auto hiddenLayerBiases1 =initLayer(1,16);
    vector<vector<double>> hiddenLayer1(1,vector<double>(16));

    auto hiddenLayerWeights2 =initLayer(16,hiddenLayer1[0].size());
    auto hiddenLayerBiases2 =initLayer(1,16);
    vector<vector<double>> hiddenLayer2( 1,vector<double>(16));

    auto outputLayerWeights =initLayer(10,hiddenLayer2[0].size());
    auto outputLayerBiases = initLayer(1,10);
    vector<vector<double>> outputLayer( 1,  vector<double>(10));

   
    const double learningRate = 0.001;

    for(int imageAt = 0; imageAt < images.size();imageAt++){

        auto image =images[imageAt];

        int label =labels[imageAt];

        
        for(int i = 0; i < hiddenLayer1[0].size(); i++){
            hiddenLayer1[0][i] = activation( hiddenLayerWeights1[i], image, hiddenLayerBiases1[0][i]);
        }

        for(int i = 0;i < hiddenLayer2[0].size();i++){
            hiddenLayer2[0][i] = activation(hiddenLayerWeights2[i], hiddenLayer1[0], hiddenLayerBiases2[0][i]);
        }

        for(int i = 0;i < outputLayer[0].size();i++){
            outputLayer[0][i] = activation(outputLayerWeights[i], hiddenLayer2[0],outputLayerBiases[0][i]);
        }

       double loss = lossFunction(outputLayer[0],label);
       int resposta = output(outputLayer[0]);

        

        vector<double> delta3 = calcOutputDelta( outputLayer[0],label);

        vector<double> delta2 = calcHiddenDelta( hiddenLayer2[0],delta3,outputLayerWeights);

        vector<double> delta1 = calcHiddenDelta( hiddenLayer1[0], delta2, hiddenLayerWeights2);

        

        updateWeights(outputLayerWeights, delta3, hiddenLayer2[0], learningRate);
        updateBiases( outputLayerBiases[0],delta3, learningRate);

      

        updateWeights(hiddenLayerWeights2, delta2,hiddenLayer1[0],learningRate);
        updateBiases(hiddenLayerBiases2[0],delta2,learningRate);

        

        updateWeights(hiddenLayerWeights1,delta1, image,learningRate);
        updateBiases(hiddenLayerBiases1[0],delta1,learningRate);

       

        if(imageAt % 1000 == 0){

            cout<< fixed<< setprecision(6);
            cout<< "\n[Imagem "<< imageAt<< "] Loss: "<< loss<< " | Pred: "<< resposta<< " | Real: "<< label;
        }
    }

    return 0;
}