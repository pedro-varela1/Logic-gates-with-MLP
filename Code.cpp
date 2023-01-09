/* MADE BY: Pedro Artur Varela, Electrical Engineering student, UFRN */

#include <math.h>
#include <cstdlib>
#include <iostream>
using namespace std;

/* Activation Function */
double sigmoid(double x) {
	return (1.0f / (1.0f + exp(-x)));
}
/* Delta Operation for the Output */
double delta_out(double guess, double target) {
	return (guess*(1-guess)*(target-guess));
}

/* Defining values for:
   Number of inputs of the network (ins),
   Number of outputs of the network (outs),
   Number of hidden neurons (hidden),
   Number of data-test (amount_data).
*/
   int const ins = 2;
   int const outs = 1;
   int const hidden = 2;
   int const amount_data = 4;

   double weights_hidden[ins][hidden]; //Hidden Layer's weights
   double weights_out[hidden][outs]; //Output Layer's weights
   double b_hidden[hidden]; //Hidden Layer's bias
   double b_out[outs]; //Output Layer's bias
   int iterations = 500000; // Training Iterations

class Perceptron
{
   /* Let's start with random values. */
   public:
   Perceptron(){
    for (int i=0; i < ins; i++){
     for (int  j=0; j < hidden; j++){
      weights_hidden [i][j] =  double((rand()%100))/100;
   }
   }
    for (int j=0; j < outs; j++){
     for (int  i=0; i < hidden; i++){
      weights_out [i][j] = double((rand()%100))/100;
       b_hidden[i] =  double((rand()%100))/100;
     }
       b_out[j] =  double((rand()%100))/100;
   }
   }
   /* Forward Propagation. */
   double guess_hidden (double inputs[amount_data][ins],int which_in, int hidden_neuron){
   double output0[amount_data][hidden];
   double sum0[amount_data][hidden] = {};
   for (int j=0; j < hidden; j++){
    for (int k=0; k < amount_data; k++){
     for (int i=0; i < ins; i++){
     sum0[k][j] = sum0[k][j] + inputs[k][i]*weights_hidden[i][j];
   }
     output0[k][j] = sigmoid (sum0[k][j] + b_hidden[j]);
   }
   }
   return output0[which_in][hidden_neuron];
   }

   double guess_out (double output0[amount_data][hidden], int which_in, int which_out){
   double output1[amount_data][outs];
   double sum1[amount_data][outs] = {};
   for (int j=0; j < outs; j++){
    for (int k=0; k < amount_data; k++){
     for (int i=0; i < hidden; i++){
     sum1[k][j] = sum1[k][j] + output0[k][i]*weights_out[i][j];
   }
     output1[k][j] = sigmoid (sum1[k][j] + b_out[j]);
   }
   }
   return output1[which_in][which_out];
   }

  double guess (double inputs[ins], int which_out){
  double out_hidden[hidden];
  double sum0[hidden] = {};
   for (int j=0; j < hidden; j++){
     for (int i=0; i < ins; i++){
     sum0[j] = sum0[j] + inputs[i]*weights_hidden[i][j];
   }
      out_hidden[j] = sigmoid (sum0[j] + b_hidden[j]);
   }
   double out[outs];
   double sum1[outs] = {};
   for (int j=0; j < outs; j++){
     for (int i=0; i < hidden; i++){
     sum1[j] = sum1[j] +  out_hidden[i]*weights_out[i][j];
   }
     out[j] = sigmoid (sum1[j] + b_out[j]);
   }
   return out[which_out];
  }

   /* Train. */
   void train (double inputs[amount_data][ins], double target[amount_data][outs]){

   for(int it=0; it < iterations; it++){
   double learning_rate = 0.01; //IMPORTANT: Defining Learning Rate.
   double shot_hidden[amount_data][hidden];
     for (int i = 0; i < hidden; i++){
      for (int j = 0; j < amount_data; j++){
       shot_hidden[j][i] = guess_hidden(inputs,j,i);
     }
     }
   double shot_out[amount_data][outs];
   double sum_sqr_error = 0;
     for (int i = 0; i < outs; i++){
      for (int j = 0; j < amount_data; j++){
       shot_out[j][i] = guess_out(shot_hidden,j,i);
        sum_sqr_error = sum_sqr_error + pow((shot_out[j][i]-target[j][i]),2);
     }
     }

    if (sum_sqr_error <= 0.001){ //Error square sum lower acceptable: stop the training.
    break;
    }

    double delta_output[amount_data][outs];
    double delta_output_b[outs];
    double dot[outs]={};
    for (int j=0; j < outs; j++){
     for (int k = 0; k < amount_data; k++){
        delta_output[k][j] = delta_out(shot_out[k][j],target[k][j]);
        dot[j] = dot[j] + (delta_output[k][j]);
   }
        delta_output_b[j] = dot[j];
   }

    double delta_layerh[amount_data][hidden];
    double dot1[amount_data][hidden]={};
    double delta_layerh_b[hidden];
    double dot2[hidden]={};
    for (int j=0; j < hidden; j++){
     for (int k = 0; k < amount_data; k++){
       for (int i=0; i < outs; i++){
        dot1[k][j] = dot1[k][j] + (delta_output[k][i]*weights_out[j][i]);
   }
        delta_layerh[k][j] = shot_hidden[k][j]*(1-shot_hidden[k][j])*dot1[k][j];
        dot2[j] = dot2[j] + (delta_layerh[k][j]);
   }
        delta_layerh_b[j] = dot2[j];
   }

    /* Hidden Layer's weights Update */
    double aux[ins][hidden] = {};
     for (int j=0; j < hidden; j++){
      for (int  i=0; i < ins; i++){
        for (int k = 0; k < amount_data; k++){
          aux[i][j] =  aux[i][j] + (delta_layerh[k][j]*inputs[k][i]);
   }
          weights_hidden [i][j] =  weights_hidden[i][j] + (learning_rate*aux[i][j]);
   }
   }

   /* Bias Update */
     for (int i=0; i < outs; i++){
       b_out[i] =  b_out[i] + (learning_rate*(delta_output_b[i]));
     }
     for (int i=0; i < hidden; i++){
      b_hidden[i] =  b_hidden[i] + (learning_rate*(delta_layerh_b[i]));
     }

   /* Output Layer's weights Update */
   double aux2[hidden][outs] = {};
     for (int j=0; j < outs; j++){
      for (int  i=0; i < hidden; i++){
       for (int k = 0; k < amount_data; k++){
         aux2[i][j] =  aux2[i][j] + (delta_output[k][j]*shot_hidden[k][i]);
   }
         weights_out [i][j] =  weights_out[i][j] + (learning_rate*aux2[i][j]);
   }
   }
   }
   }
};

int main (){
Perceptron brain;

     double in[amount_data][ins]={{0,0},{0,1},{1,0},{1,1}}; // All possibles amount_data-sets.
     double out[amount_data][outs]={{0},{1},{1},{0}}; //Outputs for each amount_data-set (XOR PROBLEM).
     brain.train(in,out);
        for (int j=0; j < amount_data; j++){
        for (int k=0; k < outs; k++){
      cout << brain.guess(in[j],k) << endl;
    }
    }

return 0;
}
