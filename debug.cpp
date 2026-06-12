#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>


#include "utils.h"


using namespace std;

double sigmoid(double x){
    return 1.0 /
     (1.0 + exp(-x));
}
double derSigmoid(double x){
    return sigmoid(x)*(1-sigmoid(x));
}

double relu(double x) {
    return max(0.0, x);
}

void derMatrix(vector<vector<double>>& matrix){
    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j < matrix[i].size(); j++){
            matrix[i][j] = derSigmoid(matrix[i][j]);    
        }
    }
}

double activation(vector<double>& weights, vector<double>& activations, double bias){
    double sum = 0;

    for(int i = 0; i < activations.size(); i++ ){
        sum += weights[i] * activations[i];
    }

    sum += bias;
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

void lossFunctionVector(vector<double>& outputLayer, int label){

    for(int i = 0; i < outputLayer.size(); i++){

        double target =
            (i == label) ? 1.0 : 0.0;

        outputLayer[i] =
            2.0 * (outputLayer[i] - target);
    }

   
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

/*vector<vector<double>> multMatrix(vector<vector<double>>& x, vector<vector<double>>& y){
    
    vector<vector<double>> temp((x.size()), vector<double>(y[0].size()));
    for(int i = 0; i<x.size();i++){ 
        for(int k = 0; k<y[0].size();k++){
            for(int j = 0; j<y.size();j++){
                
                temp[i][j] += y[k][j] * x[i][k]; 
            }
        }
    }
    return temp;
}*/

vector<vector<double>> multMatrix(vector<vector<double>>& x,vector<vector<double>>& y){

    vector<vector<double>> temp(x.size(),vector<double>(y[0].size(),0.0));

    for(int i = 0; i < x.size(); i++){
        for(int j = 0; j < y[0].size(); j++){
            for(int k = 0; k < y.size(); k++){
                temp[i][j] += x[i][k] * y[k][j];
            }
        }
    }

    return temp;
}


void backPropagation(vector<vector<double>> lastLayer, vector<vector<double>>& atualLayer){
    derMatrix(lastLayer);
    

    int linhas = min((int)atualLayer.size(),(int)lastLayer.size());

    for(int i = 0; i < linhas; i++){

        int colunas = min((int)atualLayer[i].size(), (int)lastLayer[i].size());

        for(int j = 0; j < colunas; j++){

            atualLayer[i][j] -= 0.001 * lastLayer[i][j];
        }
    }
}


int main() {
    
    auto images = readImages("dataset/train-images.idx3-ubyte");
    auto labels = readLabels("dataset/train-labels.idx1-ubyte");

        //TODO: mudar tudo esse caralho pra std array [] (mto serviço nuuuuuuuuuuuuuu)
        auto hiddenLayerWeights1 = initLayer(16, images[0].size());
        auto hiddenLayerBiases1 = initLayer(1,16);
        vector<vector<double>> hiddenLayer1((1),vector<double>(16));

        auto hiddenLayerWeights2 = initLayer(16, hiddenLayer1[0].size());
        auto hiddenLayerBiases2 = initLayer(1,16);
        vector<vector<double>> hiddenLayer2((1),vector<double>(16));

        auto outputLayerWeights =  initLayer(10, hiddenLayer2[0].size());
        auto outputLayerBiases = initLayer(1,10);
        vector<vector<double>> outputLayer((1),vector<double>(10));

        cout << "Imagens carregadas: " << images.size() << " | Labels carregadas: " << labels.size() << endl;

        for(int imageAt = 0; imageAt < images.size(); imageAt++){

            auto image = images[imageAt];
            auto label = labels[imageAt];

            for(int i = 0; i < hiddenLayer1[0].size(); i++){
                hiddenLayer1[0][i] = activation(hiddenLayerWeights1[i], image, hiddenLayerBiases1[0][i]);
            }

            for(int i = 0; i < hiddenLayer2[0].size(); i++){
                hiddenLayer2[0][i] = activation(hiddenLayerWeights2[i], hiddenLayer1[0], hiddenLayerBiases2[0][i]);
            }

            for(int i = 0; i < outputLayer[0].size(); i++){
                outputLayer[0][i] = activation(outputLayerWeights[i], hiddenLayer2[0], outputLayerBiases[0][i]);
            }



            double loss = lossFunction(outputLayer[0],label);
            int resposta = output(outputLayer[0]);
            lossFunctionVector(outputLayer[0],label);


            backPropagation(outputLayer, outputLayerWeights);



            backPropagation(multMatrix(outputLayer, outputLayerWeights), hiddenLayerWeights2);
            backPropagation(multMatrix(hiddenLayer2, hiddenLayerWeights2), hiddenLayerWeights1);



            if (imageAt % 1000 == 0 || imageAt == 0) {
                cout << fixed << setprecision(6);
                cout << "\n[Imagem n: " << imageAt << "] Perda: " << loss 
                    << " | Resposta Rede: " << resposta << " | Alvo Real: " << label;
            }
        }
    
    return 0;
}