#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int run_automata(int **t_m,int *in,int n_state,int n_sigma,int init_state,int final_state,int n){
	int i;
	int state = init_state;
	for(i=0;i<n;i++){
		//printf("%d ",state);
		state = t_m[state][in[i]];
	}
	if(state==final_state){
		return 1;
	}
	else{
		return 0;
	}
}

int main(){
	//Variables
	int STATES,SIGMA,INITIAL_STATE,FINAL_STATE,INPUT_LENGTH;
	int i,j;
	clock_t startTime;
	//Taking input
	//cin >> STATES >> SIGMA >> FINAL_STATE >> INPUT_LENGTH;
	scanf("%d %d %d %d",&STATES,&SIGMA,&FINAL_STATE,&INPUT_LENGTH);
	//An additional state has to be added for complete transition function
	STATES++;
	//Input memory allocation and input retrival
	int *input = (int *)malloc(sizeof(int)*INPUT_LENGTH);
	for(i=0;i<INPUT_LENGTH;i++){
		scanf("%d",&input[i]);
	}
	//Allocating memory and retriving to transition matrix
	int **transition_matrix = (int **)malloc(sizeof(int *)*STATES);
	int *transition_matrix_data = (int *)malloc(sizeof(int)*STATES*SIGMA);
	for(i=0;i<STATES;i++){
		transition_matrix[i] = &transition_matrix_data[i*SIGMA];
	}

	for(i=0;i<STATES;i++){
		for(j=0;j<SIGMA;j++){
			scanf("%d",&transition_matrix[i][j]);
		}
	}
	//printing the input taken
	/*for(i=0;i<INPUT_LENGTH;i++){
		printf("%d ",input[i]);
	}*/
	printf("\n");
	for(i=0;i<STATES;i++){
		for(j=0;j<SIGMA;j++){
			printf("%d ",transition_matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	//////////////////////////////////////////////////////////////////////////
	startTime = clock();
	if(run_automata(transition_matrix,input,STATES,SIGMA,INITIAL_STATE,FINAL_STATE,INPUT_LENGTH)){
		printf("String is accepted by Automata\n");
	}
	else{
		printf("String is not accepted ny Automata\n");
	}
	printf("Time for sequential DFA is %lf\n",(double)(( clock() - startTime ) / (double)CLOCKS_PER_SEC));

	return 0;
}