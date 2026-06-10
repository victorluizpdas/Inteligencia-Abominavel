#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iomanip> // Used for alignment and decimals instead of std::format

#include <opencv2/opencv.hpp>

#include "utils.h"

using namespace std;

// Keeps outputs bounded between 0 and 1
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// TODO: ReLu implementation for hidden layers if you want to swap it out later:
double relu(double x) {
    return max(0.0, x);
}

double normalizeNeuron(double x){
    return sigmoid(x);
}

// Fixed bug: Loop matches incoming activations size; weights must be size activations.size() + 1
double activation(const vector<double>& weights, const vector<double>& activations){
    double sum = 0;

    for(size_t i = 0; i < activations.size(); i++ ){
        sum += weights[i] * activations[i];
    }

    // The very last element is treated as the bias
    sum += weights[activations.size()];
    return normalizeNeuron(sum);
}

// Debug function to inspect what the neurons are doing inside a layer using standard cout streaming
void printLayerState(const string& layerName, const vector<vector<double>>& layerWeights, const vector<double>& inputs, const vector<double>& outputs) {
    cout << "\n--- Debugging " << layerName << " ---\n";
    
    // Set formatting for numbers
    cout << fixed << setprecision(3);
    
    for(size_t i = 0; i < outputs.size(); i++) {
        double sumWithoutBias = 0;
        for(size_t j = 0; j < inputs.size(); j++) {
            sumWithoutBias += layerWeights[i][j] * inputs[j];
        }
        double bias = layerWeights[i][inputs.size()];
        
        cout << "  Neuron " << setw(2) << i 
             << ": Sum = " << setw(8) << sumWithoutBias 
             << " | Bias = " << setw(8) << bias 
             << " | Activation = " << setprecision(4) << outputs[i] << "\n";
    }
}

double lossFunction(const vector<double>& outputLayer, int label){
    double sum = 0;

    for(size_t i = 0; i < outputLayer.size(); i++){
        if(static_cast<int>(i) == label){
            sum += pow(1.0 - outputLayer[i], 2);
        }else{
            sum += pow(0.0 - outputLayer[i], 2);
        }
    }

    return sum;
}

int output(const vector<double>& outputLayer){
    double maxVal = outputLayer[0];
    int resposta = 0;

    for(size_t i = 1; i < outputLayer.size(); i++){
        if(outputLayer[i] > maxVal){
            maxVal = outputLayer[i];
            resposta = i;
        }
    }

    return resposta;
}

int main() {
    
    auto images = readImages("dataset/train-images.idx3-ubyte");
    auto labels = readLabels("dataset/train-labels.idx1-ubyte");

    if (images.empty() || labels.empty()) {
        cerr << "Error: Dataset failed to load. Check file paths.\n";
        return 1;
    }

    // Explicitly allocating weights + 1 slot extra for the bias element
    auto hiddenLayerWeights1 = initLayer(16, images[0].size() + 1);
    vector<double> hiddenLayer1(16);

    auto hiddenLayerWeights2 = initLayer(16, hiddenLayer1.size() + 1);
    vector<double> hiddenLayer2(16);

    auto outputLayerWeights =  initLayer(10, hiddenLayer2.size() + 1);
    vector<double> outputLayer(10);

    for(size_t imageAt = 0; imageAt < images.size(); imageAt++){

        const auto& image = images[imageAt];
        int label = labels[imageAt];

        // 1. Forward Pass: Hidden Layer 1
        for(size_t i = 0; i < hiddenLayer1.size(); i++){
            hiddenLayer1[i] = activation(hiddenLayerWeights1[i], image);
        }

        // 2. Forward Pass: Hidden Layer 2
        for(size_t i = 0; i < hiddenLayer2.size(); i++){
            hiddenLayer2[i] = activation(hiddenLayerWeights2[i], hiddenLayer1);
        }

        // 3. Forward Pass: Output Layer
        for(size_t i = 0; i < outputLayerWeights.size(); i++){
            outputLayer[i] = activation(outputLayerWeights[i], hiddenLayer2);
        }

        // Detailed Terminal Debugging (Only runs for the first image to avoid terminal spam)
        if (imageAt == 0) {
            cout << "\n====== INITIAL DETAILED NEURON VISUALIZATION (Image 0) ======";
            printLayerState("Hidden Layer 1", hiddenLayerWeights1, image, hiddenLayer1);
            printLayerState("Hidden Layer 2", hiddenLayerWeights2, hiddenLayer1, hiddenLayer2);
            printLayerState("Output Layer", outputLayerWeights, hiddenLayer2, outputLayer);
            cout << "==============================================================\n";
        }

        double loss = lossFunction(outputLayer, label);
        int resposta = output(outputLayer);
        
        // Regular progress tracker
        if (imageAt % 1000 == 0 || imageAt == 0) {
            cout << fixed << setprecision(6);
            cout << "\n[Imagem n: " << imageAt << "] Perda: " << loss 
                 << " | Resposta Rede: " << resposta << " | Alvo Real: " << label;
        }
    }

    return 0;
}