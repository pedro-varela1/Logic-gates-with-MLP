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
/* Delta Operation for the Hidden Neuron */
double delta_hidden(double y, double w, double delta) {
	return (y*(1-y)*(w)*(delta));
}
/* Defining values for:
   Number of inputs of the network (ins),
   Number of outputs of the network (outs),
   Number of hidden neurons (hidden),
   Number of data-test (data).
*/
   const int ins = 2;
   const int outs = 1;
   const int hidden = 2;
   const int data = 4;

   double weights_hidden[ins][hidden][data]; //Hidden Layer's weights
   double weights_out[hidden][outs][data]; //Output Layer's weights
   double b_hidden[hidden][data]; //Hidden Layer's bias
   double b_out[outs][data]; //Output Layer's bias
   int iterations = 10000; // Training Iterations

class Perceptron
{
   /* Let's start with random values. */
   public:
   Perceptron(){
   for (int k=0; k < data; k++){
    for (int i=0; i < ins; i++){
     for (int  j=0; j < hidden; j++){
      weights_hidden [i][j][k] =  double((rand()%100))/100;
   }
   }
   }
   for (int k=0; k < data; k++){
    for (int j=0; j < outs; j++){
     for (int  i=0; i < hidden; i++){
      weights_out [i][j][k] = double((rand()%100))/100;
       b_hidden[i][k] =  double((rand()%100))/100;
     }
       b_out[j][k] =  double((rand()%100))/100;
   }
   }
   }
   /* Forward Propagation. */
   double guess_hidden (double inputs[data][ins],int which_in, int hidden_neuron){
   double output0[data][hidden];
   double sum0[data][hidden] = {};
   for (int j=0; j < hidden; j++){
    for (int k=0; k < data; k++){
     for (int i=0; i < ins; i++){
     sum0[k][j] = sum0[k][j] + inputs[k][i]*weights_hidden[i][j][k];
   }
      output0[k][j] = sigmoid (sum0[k][j] + b_hidden[j][k]);
   }
   }
   return output0[which_in][hidden_neuron];
   }

   double guess_out (double output0[data][hidden], int which_in, int which_out){
   double output1[data][outs];
   double sum1[data][outs] = {};
   for (int j=0; j < outs; j++){
    for (int k=0; k < data; k++){
     for (int i=0; i < hidden; i++){
     sum1[k][j] = sum1[k][j] + output0[k][i]*weights_out[i][j][k];
   }
     output1[k][j] = sigmoid (sum1[k][j] + b_out[j][k]);
   }
   }
   return output1[which_in][which_out];
   }

  double guess (double inputs[data][ins], int which_in, int which_out){
  double out_hidden[data][hidden];
     for (int i = 0; i < hidden; i++){
      for (int j = 0; j < data; j++){
       out_hidden[j][i] = guess_hidden(inputs,j,i);
     }
     }
   double out[data][outs];
     for (int i = 0; i < outs; i++){
      for (int j = 0; j < data; j++){
       out[j][i] = guess_out(out_hidden,j,i);
     }
     }
    return out[which_in][which_out];
  }

   /* Train. */
   void train (double inputs[data][ins], double target[data][outs]){
   for(int it=0; it < iterations; it++){
   double learning_rate = 0.05; //IMPORTANT: Defining Learning Rate.
   double shot_hidden[data][hidden];
     for (int i = 0; i < hidden; i++){
      for (int j = 0; j < data; j++){
       shot_hidden[j][i] = guess_hidden(inputs,j,i);
     }
     }
   double shot_out[data][outs];
   double sum_sqr_error = 0;
     for (int i = 0; i < outs; i++){
      for (int j = 0; j < data; j++){
       shot_out[j][i] = guess_out(shot_hidden,j,i);
        sum_sqr_error = sum_sqr_error + pow((shot_out[j][i]-target[j][i]),2);
     }
     }

    if (sum_sqr_error <= 0.01){ //Error square sum lower acceptable: stop the training.
    break;
    }

    double delta_output[data][outs];
    for (int j=0; j < outs; j++){
     for (int k = 0; k < data; k++){
        delta_output[k][j] = delta_out(shot_out[k][j],target[k][j]);
   }
   }
    double delta_layerh[data][hidden];
    for (int j=0; j < hidden; j++){
    for (int i=0; i < outs; i++){
     for (int k = 0; k < data; k++){
        delta_layerh[k][j] = delta_hidden(shot_hidden[k][j],weights_out[j][i][k],delta_output[k][i]);
   }
   }
   }

    /* Hidden Layer's weights Update */
     for (int j=0; j < hidden; j++){
      for (int  i=0; i < ins; i++){
        for (int k = 0; k < data; k++){
          weights_hidden [i][j][k] =  weights_hidden[i][j][k] + (learning_rate*(delta_layerh[k][j]*inputs[k][i]));
   }
   }
     }
   /* Bias Update */
     for (int i=0; i < outs; i++){
      for (int j = 0; j < data; j++){
       b_out[i][j] =  b_out[i][j] + (learning_rate*(delta_output[j][i]));
     }
     }
     for (int i=0; i < hidden; i++){
      for (int j=0; j < data; j++){
      b_hidden[i][j] =  b_hidden[i][j] + (learning_rate*(delta_layerh[j][i]));
     }
     }
   /* Output Layer's weights Update */
     for (int j=0; j < outs; j++){
      for (int  i=0; i < hidden; i++){
       for (int k = 0; k < data; k++){
         weights_out [i][j][k] =  weights_out[i][j][k] + (learning_rate*(delta_output[k][j]*target[k][j]));
   }
   }
   }
   }

   }
};

int main (){
Perceptron brain;

     double in[data][ins]={{0,0},{0,1},{1,0},{1,1}}; // All possibles data-sets.
     double out[data][outs]={{0},{1},{1},{0}}; //Outputs for each data-set (XOR PROBLEM).
     brain.train(in,out);

     for (int i =0; i < data; i++){
      for (int j=0; j  < outs; j++){
      cout << brain.guess(in,i,j) << endl;
     }
     }

return 0;
}
