
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>

using namespace std;

int **state;
int *str;
int state0;
int **pt;
int n2,num1;
class node{
	public:
		int nrow;
		int ncol;
		node(int n=0,int m=0):nrow(n),ncol(m){};
};
void create_inputfile(int n_states,int n_alphabet,int n_input,int f_s){
	ofstream outfile;
	outfile.open("input_file_random.txt");
	outfile<<n_states<<" "<<n_alphabet<<endl;
	outfile<<f_s<<" "<<n_input<<endl;
	int k;
	for(int i=0;i<n_input-1;i++){
		k=rand()%n_alphabet;
		outfile<<k<<" ";
	}
	k=rand()%n_alphabet;
	outfile<<k<<endl;

	for(int i=0;i<n_states;i++){
		for(int j=0;j<n_alphabet-1;j++){
			k=rand()%(n_states+1);
			outfile<<k<<" ";
		}
		k=rand()%(n_states+1);
		outfile<<k<<endl;
	}
	k=n_states;
	for(int j=0;j<n_alphabet-1;j++){
			outfile<<k<<" ";
		}
		outfile<<k;
	outfile.close();
}

int main(){
	int n=4,a=5,m=pow(2,27);
	int final_state = 3;
	create_inputfile(n,a,m,final_state);
	return 0;
}