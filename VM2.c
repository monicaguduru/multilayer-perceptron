
/*Mean squared deviation*/
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>

#define eta 0.1
#define epsilon 0.01
double sigmoidFunction(double x){
	return 1.0/(1.0+exp(-x));
}

double sigmoidFunctionDerivative(double x){
	return sigmoidFunction(x) * (1-sigmoidFunction(x));
}

void generateRandom(double innerW[][9], double outputW[][11], int length){
	srand ( time(NULL) );
	int i, j;
	for(i=0;i<17;i++){
		for(j=0;j<length;j++){
            innerW[i][j] =(1+rand()%99)/100.0;			
        }
    }
    for(i=0;i<=length;i++){
		for(j=0;j<10;j++){
            outputW[i][j] = (1+rand()%99)/100.0;			
        }
	}
}

void readTraining(double inputs[][17], double target[]){
    FILE* fp = fopen("train1.txt", "r");
    int i,j;
    while(getc(fp) != EOF){
        for(i=0;i<2216;i++){
            for(j=0;j<17;j++){
                if(j == 0){
                    fscanf(fp,"%lf ",&target[i]);
                    inputs[i][j] = 1;
                }
                else{
                    fscanf(fp,"%lf ",&inputs[i][j]);
                }
            }
        }
    }
    fclose(fp);
}
void readTest(double inputs[][17], double target[]){
    FILE* fp = fopen("test.txt", "r");
    int i,j;
    while(getc(fp) != EOF){
        for(i=0;i<2216;i++){
            for(j=0;j<17;j++){
                if(j == 0){
                    fscanf(fp,"%lf ",&target[i]);
                    inputs[i][j] = 1;
                }
                else{
                    fscanf(fp,"%lf ",&inputs[i][j]);
                }
            }
        }
    }
    fclose(fp);
}


void calculateOutput(double inputs[], double innerW[][9], double outputW[][11], int noOfHiddenNeurons, double innerDot[], double layer1[], double outputDot[], double finalAnswer[]){
	int i, j, k;
	double ans;
	for(i=0;i<=noOfHiddenNeurons;i++){
	     ans = 0;
	    for(j=0;j<=17;j++){
            ans += (inputs[j]*innerW[j][i]);
	}
        innerDot[i] = ans;
        layer1[i] = sigmoidFunction(innerDot[i]);
    }
    for(i=0;i<10;i++){
		 ans = 0;
		for(j=0;j<=noOfHiddenNeurons;j++){
			ans += (layer1[j] * outputW[j][i]);
		}
		outputDot[i] = ans;
        finalAnswer[i] = sigmoidFunction(outputDot[i]);
    }
}

void updateErrors(double finalAnswer[], double target, double errors[]){
    int i;
    for(i=0;i<10;i++){
        if(i+1==target){
            errors[i]=1-finalAnswer[i];
}
        else{
            errors[i]=-1*finalAnswer[i];
}
    }
}

int updateWeights(int cond, double errors[],double outputW[][11],double layer1[],double outputDot[],int noOfHiddenNeurons,double innerDot[] ,double innerW[][9], double X[]){
	int i, j;
	double delta1[10], delta2[9];
   
	for(i=0;i<10;i++){
		delta1[i] = -1 * errors[i] * sigmoidFunctionDerivative(outputDot[i]); 
    }
    for(i=0;i<noOfHiddenNeurons;i++){
        delta2[i]=0;
        for(j=0;j<10;j++){
            delta2[i] += delta1[j] * outputW[i][j] * sigmoidFunctionDerivative(innerDot[i]);
        }
    }
	for(i=0;i<=noOfHiddenNeurons;i++){
		for(j=0;j<10;j++){
            double deltaW = eta * layer1[i] * delta1[j];
            outputW[j][i] -= deltaW;
            if(cond==2 && abs(deltaW) < epsilon)
                return 1;
		}       
    }
	
	for(i=0;i<=17;i++){
    		for(j=0;j<=noOfHiddenNeurons;j++){
			innerW[i][j] -= eta * delta2[j] * X[i];
		}		
    }
}	
int classifyClass(double finalAnswer[]){
	int i=0,maxi=0;
	for(i=0;i<10;i++){
		if(finalAnswer[maxi]<finalAnswer[i]){
			maxi=i;
		}
	}
	return maxi;
}
void train(int cond, double inputs[][17], int size, double innerW[][9], double outputW[][11], int noOfHiddenNeurons, double innerDot[], double layer1[], double outputDot[], double finalAnswer[], double target[]){
    int i, j;
	double errors[10];
    for(i=0;i<size;i++){
        j=0;
        do{
		    calculateOutput(inputs[i], innerW, outputW, noOfHiddenNeurons, innerDot, layer1, outputDot, finalAnswer);	
            	updateErrors(finalAnswer, target[i], errors);
		    if(updateWeights(cond, errors, outputW, layer1, innerDot, noOfHiddenNeurons, finalAnswer, innerW, inputs[i])){
                break;
            }
            j++;
            if (j==100 && cond==1)
                break;
	    }
        while(1);
    }
}
void test(double testInput[][17], int size, double innerW[][9], double outputW[][11], int noOfHiddenNeurons, double innerDot[], double layer1[], double outputDot[], double finalAnswer[], double target[]){
    int i, count1=0;
    for(i=0;i<size;i++){
            calculateOutput(testInput[i], innerW, outputW, noOfHiddenNeurons, innerDot, layer1, outputDot, finalAnswer);
            if( abs(target[i]-classifyClass(finalAnswer)+1)==0){
                count1++;
            }
    }
    printf("No.of Neurons:%d Accuracy:%f\n",noOfHiddenNeurons, (count1/999.0)*100);
}
int main(){
    int i, j, n;
    double inputs[2217][17], target[2217], innerW[17][9], outputW[9][11], testInput[2217][17], testTarget[2217],innerDot[17], layer1[9], outputDot[10], finalAnswer[10];
    readTraining(inputs, target);
    readTest(testInput, testTarget);
    for(j=1;j<=2;j++){
        if(j==1){
            printf("For 100 Epochs:\n");
        }
        else{
            printf("Stopping criteria:\n");
        }
        generateRandom(innerW,outputW,8);
        for(i=5;i<=8;i++){
            train(j, inputs, 2217, innerW, outputW, i, innerDot, layer1, outputDot, finalAnswer, target);
            test(testInput, 999, innerW, outputW, i, innerDot, layer1, outputDot, finalAnswer, testTarget);
        }
    }
    return 0;
}
